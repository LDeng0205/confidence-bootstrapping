import os
import pickle
from functools import partial
import copy
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdb import PDBSidechain, read_strings_from_txt
from confidence.dataset import get_args_and_cache_path, ListDataset
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.utils import get_model


class BootstrappingDataset(Dataset):
    def __init__(self, transform, cache_path, original_model_dir, model_ckpt, confidence_model_dir,
                 confidence_model_ckpt, device, sigma_schedule, limit_complexes,
                 inference_steps, inf_sched_alpha, inf_sched_beta, rot_inf_sched_alpha, rot_inf_sched_beta,
                 tor_inf_sched_alpha, tor_inf_sched_beta, samples_per_complex, different_schedules, all_atoms,
                 args, multiplicity=1, cache_ids_to_combine= None, cache_creation_id=None, iterations=1,
                 confidence_cutoff=0, fixed_length=None, max_per_complex=1, temperature=0.0,
                 ):

        super(BootstrappingDataset, self).__init__(cache_path, transform)
        self.device, self.sigma_schedule = device, sigma_schedule
        self.inference_steps = inference_steps
        self.inf_sched_alpha, self.inf_sched_beta = inf_sched_alpha, inf_sched_beta
        self.rot_inf_sched_alpha, self.rot_inf_sched_beta = rot_inf_sched_alpha, rot_inf_sched_beta
        self.tor_inf_sched_alpha, self.tor_inf_sched_beta = tor_inf_sched_alpha, tor_inf_sched_beta
        self.different_schedules, self.limit_complexes = different_schedules, limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.multiplicity = multiplicity
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args = args
        self.iterations = iterations
        self.confidence_cutoff = confidence_cutoff
        self.max_per_complex = max_per_complex
        self.confidence_model_dir = confidence_model_dir
        self.confidence_model_ckpt = confidence_model_ckpt
        self.fixed_length = fixed_length

        self.original_model_args, _ = get_args_and_cache_path(original_model_dir, '')
        self.confidence_model_args, _ = get_args_and_cache_path(confidence_model_dir, '')
        self.full_cache_path = os.path.join(cache_path,
                                            f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            + f'_limit_{limit_complexes}_ccutoff_{confidence_cutoff}')

        print("looking for generated complexes at", os.path.join(self.full_cache_path, f"complexes_id{self.cache_creation_id}.pkl"))
        if (not os.path.exists(os.path.join(self.full_cache_path, "complexes.pkl")) and self.cache_creation_id is None) or \
                (not os.path.exists(os.path.join(self.full_cache_path, f"complexes_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            print(os.path.join(self.full_cache_path, f"complexes_id{self.cache_creation_id}.pkl"), "does not exist")
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading complexes from: {os.path.join(self.full_cache_path, "complexes.pkl")}')
            with open(os.path.join(self.full_cache_path, "complexes.pkl"), 'rb') as f:
                self.complex_graphs = pickle.load(f)
        else:
            self.complex_graphs = []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                with open(os.path.join(self.full_cache_path, f"complexes_id{cache_id}.pkl"), 'rb') as f:
                    self.complex_graphs.extend(pickle.load(f))
        print('Number of complex graphs: ', len(self.complex_graphs))

        confidences = np.asarray([complex_graph.confidence for complex_graph in self.complex_graphs])
        confidences = np.clip(confidences, -5, 5)
        weights = np.exp(confidences * temperature)
        self.weights = weights / np.sum(weights)

    def len(self):
        return len(self.complex_graphs) * self.multiplicity if self.fixed_length is None else self.fixed_length

    def get(self, idx):
        idx = np.random.choice(len(self.complex_graphs), p=self.weights)
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        complex_graph = complex_graph.to_data_list()[0]
        complex_graph['ligand'].orig_pos = complex_graph['ligand'].pos.cpu().numpy()
        
        for a in ['confidence', 'random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq',
                  'to_keep', 'chain_ids', 'smiles']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)
            if hasattr(complex_graph['ligand'], a):
                delattr(complex_graph['ligand'], a)

        return complex_graph

    def preprocessing(self):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        confidence_model = get_model(self.confidence_model_args, self.device, t_to_sigma=t_to_sigma, confidence_mode=True, no_parallel=True)
        confidence_state_dict = torch.load(f'{self.confidence_model_dir}/{self.confidence_model_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(confidence_state_dict, strict=True)
        confidence_model = confidence_model.to(self.device)
        confidence_model.eval()

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

        sequences_to_embeddings = None
        if self.args.pdbsidechain_esm_embeddings_path is not None:
            print('Loading ESM embeddings')
            id_to_embeddings = torch.load(self.args.pdbsidechain_esm_embeddings_path)
            sequences_list = read_strings_from_txt(self.args.pdbsidechain_esm_embeddings_sequences_path)
            sequences_to_embeddings = {}
            for i, seq in enumerate(sequences_list):
                if str(i) in id_to_embeddings:
                    sequences_to_embeddings[seq] = id_to_embeddings[str(i)]

        dataset = PDBSidechain(root=self.args.pdbsidechain_dir, sequences_to_embeddings=sequences_to_embeddings,
                               vandermers_extraction=False, add_random_ligand=True,
                               transform=None, limit_complexes=self.args.limit_complexes,
                               receptor_radius=self.original_model_args.receptor_radius,
                               cache_path=self.args.cache_path,
                               remove_hs=self.original_model_args.remove_hs,
                               c_alpha_max_neighbors=self.original_model_args.c_alpha_max_neighbors,
                               all_atoms=self.original_model_args.all_atoms if 'all_atoms' in self.original_model_args else False,
                               atom_radius=self.original_model_args.atom_radius if 'all_atoms' in self.original_model_args else None,
                               atom_max_neighbors=self.original_model_args.atom_max_neighbors if 'all_atoms' in self.original_model_args else None,
                               num_workers=self.args.num_workers,
                               knn_only_graph=True if not hasattr(self.args, 'not_knn_only_graph') else not self.args.not_knn_only_graph)

        confidence_dataset = PDBSidechain(root=self.args.pdbsidechain_dir, sequences_to_embeddings=sequences_to_embeddings,
                                           vandermers_extraction=False, add_random_ligand=True,
                                           transform=None, limit_complexes=self.args.limit_complexes,
                                           receptor_radius=self.confidence_model_args.receptor_radius,
                                           cache_path=self.args.cache_path,
                                           remove_hs=self.confidence_model_args.remove_hs,
                                           c_alpha_max_neighbors=self.confidence_model_args.c_alpha_max_neighbors,
                                           all_atoms=self.confidence_model_args.all_atoms if 'all_atoms' in self.confidence_model_args else False,
                                           atom_radius=self.confidence_model_args.atom_radius if 'all_atoms' in self.confidence_model_args else None,
                                           atom_max_neighbors=self.confidence_model_args.atom_max_neighbors if 'all_atoms' in self.confidence_model_args else None,
                                           num_workers=self.args.num_workers,
                                           knn_only_graph=True if not hasattr(self.args, 'not_knn_only_graph') else not self.args.not_knn_only_graph)

        filtered_complexes = []
        for _ in range(self.iterations):
            complex_graphs = [dataset.get(i) for i in range(len(dataset))]
            dataset = ListDataset(complex_graphs)
            loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

            for idx, orig_complex_graph in tqdm(enumerate(loader)):
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
                randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)

                try:
                    confidence_graph = confidence_dataset.get(protein=orig_complex_graph.name[0], smiles=orig_complex_graph['ligand'].smiles[0])
                except Exception as e:
                    print("skipping", e)
                    continue

                confidence_data_list = [copy.deepcopy(confidence_graph) for _ in range(self.samples_per_complex)]

                try:
                    r = sampling(data_list=data_list, model=model, inference_steps=self.inference_steps,
                                 tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                 device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args,
                                 confidence_model=confidence_model, filtering_data_list=confidence_data_list,
                                 filtering_model_args=self.confidence_model_args)
                    predictions_list, confidences = r
                except Exception as e:
                    print("error", e)
                    continue

                conf = confidences.cpu().numpy()
                conf = np.nan_to_num(conf, nan=-1e-6)
                re_order = np.argsort(conf)[::-1]
                predictions_list = [predictions_list[i] for i in re_order]
                print(conf[re_order])
                for idx, (pred, conf) in enumerate(zip(predictions_list, conf[re_order])):
                    if conf < self.confidence_cutoff or idx >= self.max_per_complex: break
                    pred.confidence = conf
                    pred = pred.cpu()
                    filtered_complexes.append(pred)

        with open(os.path.join(self.full_cache_path, f"complexes{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump(filtered_complexes, f)
