import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.moad import MOAD
from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, crop_beyond
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}_chainCutoff{args.chain_cutoff}'
                              + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                              + ('' if args.no_torsion or args.num_conformers == 1 else f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings')
                              + ('' if args.not_full_dataset else f'_full')
                              + '_fixedKNNonly_chainOrd')
    return cache_path

def get_args_and_cache_path(original_model_dir, split, protein_file=None):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
        if not hasattr(model_args, 'separate_noise_schedule'):  # exists for compatibility
            model_args.separate_noise_schedule = False
        if not hasattr(model_args, 'lm_embeddings_path'):  # exists for compatibility
            model_args.lm_embeddings_path = None
        if not hasattr(model_args, 'all_atoms'):  # exists for compatibility
            model_args.all_atoms = False
        if not hasattr(model_args,'tr_only_confidence'):  # exists for compatibility
            model_args.tr_only_confidence = True
        if not hasattr(model_args,'high_confidence_threshold'):  # exists for compatibility
            model_args.high_confidence_threshold = 0.0
        if not hasattr(model_args, 'include_confidence_prediction'):  # exists for compatibility
            model_args.include_confidence_prediction = False
        if not hasattr(model_args, 'confidence_dropout'):
            model_args.confidence_dropout = model_args.dropout
        if not hasattr(model_args, 'confidence_no_batchnorm'):
            model_args.confidence_no_batchnorm = False
        if not hasattr(model_args, 'confidence_weight'):
            model_args.confidence_weight = 1
        if not hasattr(model_args, 'asyncronous_noise_schedule'):
            model_args.asyncronous_noise_schedule = False
        if not hasattr(model_args, 'correct_torsion_sigmas'):
            model_args.correct_torsion_sigmas = False
        if not hasattr(model_args, 'not_full_dataset'):
            model_args.not_full_dataset = True
        if not hasattr(model_args, 'esm_embeddings_path'):
            model_args.esm_embeddings_path = None
        if protein_file is not None:
            model_args.protein_file = protein_file

    return model_args, get_cache_path(model_args,split)


class FilteringDataset(Dataset):
    def __init__(self, cache_path, original_model_dir, split, device, sigma_schedule, limit_complexes,
                 inference_steps, inf_sched_alpha, inf_sched_beta, rot_inf_sched_alpha, rot_inf_sched_beta,
                 tor_inf_sched_alpha, tor_inf_sched_beta, samples_per_complex, different_schedules, all_atoms,
                 args, model_ckpt, dataset, balance=False, multiplicity=1,use_original_model_cache=True, rmsd_classification_cutoff=2,
                 atom_rmsd_classification_cutoff=2, parallel=1, cache_ids_to_combine= None, cache_creation_id=None,
                 trajectory_sampling=False, include_miscellaneous_atoms=False, atom_confidence_loss_weight=0.0,
                 crop_beyond=None, rmsd_classification_upper=None):

        super(FilteringDataset, self).__init__()
        self.device, self.sigma_schedule = device, sigma_schedule
        self.inference_steps = inference_steps
        self.inf_sched_alpha, self.inf_sched_beta = inf_sched_alpha, inf_sched_beta
        self.rot_inf_sched_alpha, self.rot_inf_sched_beta = rot_inf_sched_alpha, rot_inf_sched_beta
        self.tor_inf_sched_alpha, self.tor_inf_sched_beta = tor_inf_sched_alpha, tor_inf_sched_beta
        self.different_schedules, self.limit_complexes = different_schedules, limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.multiplicity = multiplicity
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.atom_rmsd_classification_cutoff = atom_rmsd_classification_cutoff
        self.parallel = parallel
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args, self.split = args, split
        self.trajectory_sampling = trajectory_sampling
        self.fixed_step = None
        self.fixed_sample = None
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.atom_confidence_loss_weight = atom_confidence_loss_weight
        self.crop_beyond = crop_beyond
        self.rmsd_classification_upper = rmsd_classification_upper
        self.dataset = dataset

        self.original_model_args, original_model_cache = get_args_and_cache_path(original_model_dir, split, protein_file=args.protein_file)
        self.full_cache_path = os.path.join(cache_path,
                                            ('' if dataset == "pdbbind" else dataset + '_')
                                            + f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            + f'_split_{split}_limit_{limit_complexes}'
                                            + ('' if args.protein_file == "protein_processed" else '_' + args.protein_file)
                                            )

        print("looking for ligand positions at", os.path.join(self.full_cache_path, "ligand_positions.pkl"))
        if (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl")) and self.cache_creation_id is None) or \
                (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            print(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl"), "does not exist")
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)

        self.complex_graphs_cache = original_model_cache if self.use_original_model_cache else get_cache_path(args, split)
        print('Using the cached complex graphs of the original model args' if self.use_original_model_cache else 'Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to filtering_train.py')
        print(self.complex_graphs_cache)
        if self.dataset == 'pdbbind':
            dataset = PDBBind(transform=None, root=args.pdbbind_dir, limit_complexes=args.limit_complexes,
                            chain_cutoff=args.chain_cutoff,
                            receptor_radius=args.receptor_radius,
                            cache_path=args.cache_path, split_path=args.split_val if split == 'val' else args.split_train,
                            remove_hs=args.remove_hs, max_lig_size=None,
                            c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                            matching=not args.no_torsion, keep_original=True,
                            popsize=args.matching_popsize,
                            maxiter=args.matching_maxiter,
                            all_atoms=args.all_atoms,
                            atom_radius=args.atom_radius,
                            atom_max_neighbors=args.atom_max_neighbors,
                            esm_embeddings_path=args.pdbbind_esm_embeddings_path,
                            require_ligand=True,
                            num_workers=args.num_workers,
                            protein_file=args.protein_file,
                            knn_only_graph=False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                            include_miscellaneous_atoms=False if not hasattr(args,'include_miscellaneous_atoms') else args.include_miscellaneous_atoms)
        elif self.dataset == 'moad':
            dataset = MOAD(transform=None, root=args.moad_dir, limit_complexes=args.limit_complexes,
                           chain_cutoff=args.chain_cutoff,
                           receptor_radius=args.receptor_radius,
                           cache_path=args.cache_path, split=self.split,
                           remove_hs=args.remove_hs, max_lig_size=None,
                           c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                           matching=not args.no_torsion, keep_original=True,
                           popsize=args.matching_popsize,
                           maxiter=args.matching_maxiter,
                           all_atoms=args.all_atoms,
                           atom_radius=args.atom_radius,
                           atom_max_neighbors=args.atom_max_neighbors,
                           esm_embeddings_path=args.moad_esm_embeddings_path,
                           esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                           require_ligand=True,
                           num_workers=args.num_workers,
                           knn_only_graph=True if not hasattr(self.args,
                                                              'not_knn_only_graph') else not self.args.not_knn_only_graph,
                           include_miscellaneous_atoms=False if not hasattr(self.args,
                                                                            'include_miscellaneous_atoms') else self.args.include_miscellaneous_atoms,
                           num_conformers=1,
                           unroll_clusters=self.args.unroll_clusters, remove_pdbbind=self.args.remove_pdbbind,
                           min_ligand_size=self.args.min_ligand_size,
                           max_receptor_size=self.args.max_receptor_size,
                           remove_promiscuous_targets=self.args.remove_promiscuous_targets,
                           no_randomness=True)

        complex_graphs = [dataset.get(i) for i in range(len(dataset))]
        self.complex_graph_dict = {d.name: d for d in complex_graphs}

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions.pkl")}')
            if trajectory_sampling:
                with open(os.path.join(self.full_cache_path, f"trajectories.pkl"), 'rb') as f:
                    self.full_ligand_positions, self.rmsds = pickle.load(f)
            else:
                with open(os.path.join(self.full_cache_path, "ligand_positions.pkl"), 'rb') as f:
                    self.full_ligand_positions, self.rmsds = pickle.load(f)
            if os.path.exists(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl")):
                with open(os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"), 'rb') as f:
                    generated_rmsd_complex_names = pickle.load(f)
            else:
                print('HAPPENING | The path, ', os.path.join(self.full_cache_path, "complex_names_in_same_order.pkl"),
                      ' does not exist. \n => We assume that means that we are using a ligand_positions.pkl where the '
                      'code was not saving the complex names for them yet. We now instead use the complex names of '
                      'the dataset that the original model used to create the ligand positions and RMSDs.')
                with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
                    original_model_complex_graphs = pickle.load(f)
                    generated_rmsd_complex_names = [d.name for d in original_model_complex_graphs]
            assert (len(self.rmsds) == len(generated_rmsd_complex_names))
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, ("trajectories_" if self.trajectory_sampling else "ligand_positions_") + str(cache_id)+ ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")):
                    print(f'The generated ligand positions with cache_id do not exist: {cache_id} --- SKIPPING')
                    continue
                    #raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
                if trajectory_sampling:
                    with open(os.path.join(self.full_cache_path, f"trajectories_id{cache_id}.pkl"), 'rb') as f:
                        full_ligand_positions, rmsds = pickle.load(f)
                else:
                    with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                        full_ligand_positions, rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set.intersection(*map(set, all_names_unsorted)))
            all_rmsds, all_full_ligand_positions, all_names = [], [], []
            for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
                intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
                all_rmsds.append((intermediate_rmsds))
                intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
                all_full_ligand_positions.append((intermediate_pos))
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=(1 if trajectory_sampling else 0)))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            generated_rmsd_complex_names = names_order
        print('Number of complex graphs: ', len(self.complex_graph_dict))
        print('Number of RMSDs and positions for the complex graphs: ', len(self.full_ligand_positions))
        print('1st position shape: ', self.full_ligand_positions[0].shape)

        print('Number of complexes with at least a nan', sum([np.isnan(self.full_ligand_positions[i]).any() for i in range(len(self.full_ligand_positions))]))
        print('Number of complexes with all nans', sum([np.isnan(self.full_ligand_positions[i]).all() for i in range(len(self.full_ligand_positions))]))

        self.positions_rmsds_dict = {}

        for name, pos, rmsd in zip(generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds):
            if np.isnan(pos).all():
                continue
            to_keep = np.logical_not(np.isnan(pos[:, 0, 0]))
            self.positions_rmsds_dict[name] = (pos[to_keep], rmsd[to_keep])

        self.dataset_names = list(set(self.positions_rmsds_dict.keys()) & set(self.complex_graph_dict.keys()))
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

        # counting positives and negatives
        pos, neg = 0, 0
        print("rmsds shape", self.positions_rmsds_dict[self.dataset_names[0]][1].shape)
        for name in self.dataset_names:
            r = self.positions_rmsds_dict[name][1]
            p = np.sum(r < 2)
            pos += p
            if rmsd_classification_upper is not None:
                n = np.sum(r > rmsd_classification_upper)
                neg += n
            else:
                neg += (len(r) - p)
        print("In total", pos, "positives and ", neg, "negatives")

        # for affinity prediction
        #df = pd.read_csv('data/INDEX_general_PL_data.2020', sep="  |//|=", comment='#', header=None,
        #                 names=['PDB code', 'resolution', 'release year', '-logKd/Ki', 'Kd/Ki', 'Kd/Ki value',
        #                        'reference ligand name', 'refef', 'ef', 'ee', 'asd'])
        #self.affinities = df.set_index('PDB code').to_dict()['-logKd/Ki']

    def len(self):
        return len(self.dataset_names) * self.multiplicity

    def get(self, idx):
        if self.multiplicity > 1: idx = idx % len(self.dataset_names)

        complex_graph = copy.deepcopy(self.complex_graph_dict[self.dataset_names[idx]])
        positions, rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]
        old_pos = torch.from_numpy(complex_graph['ligand'].orig_pos - complex_graph.original_center.cpu().numpy())
        t = 0

        if self.parallel > 1:
            if self.parallel == len(rmsds):
                idxs = np.arange(self.parallel)
            elif self.parallel < len(rmsds):
                idxs = np.random.choice(len(rmsds), size=self.parallel, replace=False)
            else:
                raise Exception("parallel size larger than sample size")

            N = complex_graph['ligand'].num_nodes
            complex_graph['ligand'].x = complex_graph['ligand'].x.repeat(self.parallel, 1)
            complex_graph['ligand'].edge_mask = complex_graph['ligand'].edge_mask.repeat(self.parallel)
            complex_graph['ligand', 'ligand'].edge_index = torch.cat([N*i+complex_graph['ligand', 'ligand'].edge_index for i in range(self.parallel)], dim=1)
            complex_graph['ligand', 'ligand'].edge_attr = complex_graph['ligand', 'ligand'].edge_attr.repeat(self.parallel, 1)
            complex_graph['ligand'].pos = torch.from_numpy(positions[idxs].reshape(-1, 3))
            complex_graph.rmsd = torch.from_numpy(rmsds[idxs]).unsqueeze(0)
            complex_graph.y = torch.from_numpy(rmsds[idxs]<2).unsqueeze(0).float()
        else:
            if self.trajectory_sampling:
                step = random.randint(0, len(positions)-1) if self.fixed_step is None else self.fixed_step
                t = step/(len(positions)-1)
                positions = positions[len(positions)-step-1]
            if self.balance:
                if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
                label = random.randint(0, 1)
                success = rmsds < self.rmsd_classification_cutoff
                n_success = np.count_nonzero(success)
                failure = np.logical_not(success) if self.rmsd_classification_upper is None else rmsds > self.rmsd_classification_upper
                n_failure = np.count_nonzero(failure)
                if (label == 0 and n_failure > 0) or (n_success == 0 and self.trajectory_sampling):
                    # sample negative complex
                    sample = random.randint(0, n_failure-1)
                    lig_pos = positions[failure][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
                else:
                    # sample positive complex
                    if n_success > 0:
                        sample = random.randint(0, n_success - 1)
                        lig_pos = positions[success][sample]
                        complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
                    else:
                        # if no successfull sample returns the matched complex but first check that that is within 2A
                        filterHs = torch.not_equal(complex_graph['ligand'].x[:, 0], 0).cpu().numpy()
                        ligand_pos = np.asarray(complex_graph['ligand'].pos.cpu().numpy()[filterHs])
                        orig_ligand_pos = np.expand_dims(complex_graph['ligand'].orig_pos[filterHs] - complex_graph.original_center.cpu().numpy(), axis=0)
                        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
                        if rmsd > 2:
                            # need to give original position
                            complex_graph['ligand'].pos = torch.from_numpy(complex_graph['ligand'].orig_pos).float()

                complex_graph.y = torch.tensor(label).float().unsqueeze(0)
            else:
                sample = random.randint(0, len(positions) - 1) if self.fixed_sample is None else self.fixed_sample
                complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
                if isinstance(self.rmsd_classification_cutoff, list):
                    complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
                    complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
                complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

        complex_graph['ligand'].node_t = {'tr': t * torch.ones(complex_graph['ligand'].num_nodes),
                                          'rot': t * torch.ones(complex_graph['ligand'].num_nodes),
                                          'tor': t * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': t * torch.ones(complex_graph['receptor'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['receptor'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': t * torch.ones(complex_graph['atom'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['atom'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['atom'].num_nodes)}
        if self.include_miscellaneous_atoms:
            complex_graph['misc_atom'].node_t = {'tr': t * torch.ones(complex_graph['misc_atom'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['misc_atom'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['misc_atom'].num_nodes)}
        complex_graph.complex_t = {'tr': t * torch.ones(1), 'rot': t * torch.ones(1), 'tor': t * torch.ones(1)}
        #complex_graph.affinity = torch.tensor(self.affinities[complex_graph.name]).float()

        if self.atom_confidence_loss_weight>0:
            assert self.parallel == 1, "not implemented yet"
            atom_distances = torch.norm(old_pos - complex_graph['ligand'].pos, dim=1)
            if isinstance(self.atom_rmsd_classification_cutoff, list):
                cutoff_tensor = torch.tensor([0.0] + self.atom_rmsd_classification_cutoff + [math.inf])
                complex_graph.atom_y_binned = \
                    torch.logical_and(atom_distances.unsqueeze(1) < cutoff_tensor[1:].unsqueeze(0),
                                      atom_distances.unsqueeze(1) >= cutoff_tensor[:-1].unsqueeze(0)).float()
            else:
                complex_graph.atom_y = (atom_distances < self.atom_rmsd_classification_cutoff).float()

        if self.crop_beyond is not None:
            crop_beyond(complex_graph, self.crop_beyond, self.all_atoms)
        return complex_graph

    def preprocessing(self, original_model_cache):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        tr_schedule = get_t_schedule(sigma_schedule=self.sigma_schedule, inference_steps=self.inference_steps,
                                     inf_sched_alpha=self.inf_sched_alpha, inf_sched_beta=self.inf_sched_beta)
        if self.different_schedules:
            rot_schedule = get_t_schedule(sigma_schedule=self.rot_sigma_schedule, inference_steps=self.inference_steps,
                                          inf_sched_alpha=self.rot_inf_sched_alpha,
                                          inf_sched_beta=self.rot_inf_sched_beta)
            tor_schedule = get_t_schedule(sigma_schedule=self.tor_sigma_schedule, inference_steps=self.inference_steps,
                                          inf_sched_alpha=self.tor_inf_sched_alpha,
                                          inf_sched_beta=self.tor_inf_sched_beta)
            print('tr schedule', tr_schedule)
            print('rot schedule', rot_schedule)
            print('tor schedule', tor_schedule)

        else:
            rot_schedule = tr_schedule
            tor_schedule = tr_schedule
            print('common t schedule', tr_schedule)

        if self.dataset == 'pdbbind':
            dataset = PDBBind(transform=None, root=self.args.pdbbind_dir, limit_complexes=self.args.limit_complexes,
                              chain_cutoff=self.args.chain_cutoff,
                              receptor_radius=self.original_model_args.receptor_radius,
                              cache_path=self.args.cache_path, split_path=self.args.split_val if self.split == 'val' else self.args.split_train,
                                remove_hs=self.original_model_args.remove_hs, max_lig_size=None,
                                c_alpha_max_neighbors=self.original_model_args.c_alpha_max_neighbors,
                                matching=not self.original_model_args.no_torsion, keep_original=True,
                                popsize=self.original_model_args.matching_popsize,
                                maxiter=self.original_model_args.matching_maxiter,
                                all_atoms=self.original_model_args.all_atoms,
                                atom_radius=self.original_model_args.atom_radius,
                                atom_max_neighbors=self.original_model_args.atom_max_neighbors,
                                esm_embeddings_path=self.args.esm_embeddings_path,
                                require_ligand=True,
                                num_workers=self.args.num_workers,
                                protein_file=self.args.protein_file,
                              knn_only_graph=False if not hasattr(self.args, 'not_knn_only_graph') else not self.args.not_knn_only_graph,
                              include_miscellaneous_atoms= False if not hasattr(self.args,'include_miscellaneous_atoms') else self.args.include_miscellaneous_atoms,
            )
        elif self.dataset == 'moad':
            dataset = MOAD(transform=None, root=self.args.moad_dir, limit_complexes=self.args.limit_complexes,
                           chain_cutoff=self.args.chain_cutoff,
                           receptor_radius=self.original_model_args.receptor_radius,
                           cache_path=self.args.cache_path, split=self.split,
                           remove_hs=self.original_model_args.remove_hs, max_lig_size=None,
                           c_alpha_max_neighbors=self.original_model_args.c_alpha_max_neighbors,
                           matching=not self.original_model_args.no_torsion, keep_original=True,
                           popsize=self.args.matching_popsize,
                           maxiter=self.args.matching_maxiter,
                           all_atoms=self.original_model_args.all_atoms if 'all_atoms' in self.original_model_args else False,
                           atom_radius=self.original_model_args.atom_radius if 'all_atoms' in self.original_model_args else None,
                           atom_max_neighbors=self.original_model_args.atom_max_neighbors if 'all_atoms' in self.original_model_args else None,
                           esm_embeddings_path=self.args.esm_embeddings_path,
                           esm_embeddings_sequences_path=self.args.moad_esm_embeddings_sequences_path,
                           require_ligand=True,
                           num_workers=self.args.num_workers,
                           knn_only_graph=True if not hasattr(self.args,
                                                              'not_knn_only_graph') else not self.args.not_knn_only_graph,
                           include_miscellaneous_atoms=False if not hasattr(self.args,
                                                                            'include_miscellaneous_atoms') else self.args.include_miscellaneous_atoms,
                           num_conformers=1,
                           unroll_clusters=self.args.unroll_clusters, remove_pdbbind=self.args.remove_pdbbind,
                           min_ligand_size=self.args.min_ligand_size,
                           max_receptor_size=self.args.max_receptor_size,
                           remove_promiscuous_targets=self.args.remove_promiscuous_targets,
                           no_randomness=True)
        complex_graphs = [dataset.get(i) for i in range(len(dataset))]
        dataset = ListDataset(complex_graphs)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        rmsds, full_ligand_positions, names, trajectories = [], [], [], []
        for idx, orig_complex_graph in tqdm(enumerate(loader)):
            # TODO try to get the molecule directly from file without and check same results to avoid any kind of leak
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list == None:
                try:
                    r = sampling(data_list=data_list, model=model, inference_steps=self.inference_steps,
                                 tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                 device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args, return_full_trajectory=self.trajectory_sampling)
                    predictions_list, confidences, trajectory = r if self.trajectory_sampling else (r[0], r[1], None)
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                            break
                        print('| WARNING: SVD failed to converge - trying again with a new sample')
                    else:
                        print("error", e)
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('skipping')
                            break

            if failed_convergence_counter > 5: continue
            if self.original_model_args.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
            orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

            rmsds.append(rmsd)
            final_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list])
            full_ligand_positions.append(final_pos)
            names.append(orig_complex_graph.name[0])
            if self.trajectory_sampling:
                trajectory.append(final_pos)
                trajectories.append(np.asarray(trajectory))
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.
        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions, rmsds), f)
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((names), f)
        with open(os.path.join(self.full_cache_path, f"trajectories{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((trajectories, rmsds), f)


class FilteringMOAD(Dataset):
    def __init__(self, split, transform, args, device, balance=False):
        
        super(FilteringMOAD, self).__init__()

        self.device = device
        self.balance = balance
        self.rmsd_classification_cutoff = args.rmsd_classification_cutoff
        self.atom_rmsd_classification_cutoff = args.atom_rmsd_classification_cutoff
        self.cache_ids_to_combine = args.cache_ids_to_combine
        self.cache_creation_id = args.cache_creation_id
        self.samples_per_complex = args.samples_per_complex
        self.args, self.split = args, split
        self.cache_path = args.cache_path

        print('|LOADING Moad dataset')

        self.dataset = MOAD(transform=transform, root=args.moad_dir, limit_complexes=args.limit_complexes,
                       chain_cutoff=args.chain_cutoff,
                       receptor_radius=args.receptor_radius,
                       cache_path=args.cache_path, split=args.split,
                       remove_hs=args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                       matching=not args.no_torsion, keep_original=True,
                       popsize=args.matching_popsize,
                       maxiter=args.matching_maxiter,
                       all_atoms=args.all_atoms,
                       atom_radius=args.atom_radius if 'all_atoms' in args else None,
                       atom_max_neighbors=args.atom_max_neighbors if 'all_atoms' in args else None,
                       esm_embeddings_path=args.esm_embeddings_path,
                       esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                       require_ligand=True,
                       num_workers=args.num_workers,
                       protein_file=args.protein_file,
                       knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                       include_miscellaneous_atoms=False if not hasattr(args,
                                                                        'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                       num_conformers=1,
                       unroll_clusters=args.unroll_clusters, remove_pdbbind=args.remove_pdbbind,
                       min_ligand_size=args.min_ligand_size,
                       max_receptor_size=args.max_receptor_size,
                       remove_promiscuous_targets=args.remove_promiscuous_targets)
        print('|SUCCES Dataset loaded')

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        complex_graph = self.dataset[idx]
        return complex_graph