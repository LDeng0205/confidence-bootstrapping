import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import subgraph

from datasets.dataloader import DataLoader, DataListLoader
from datasets.moad import MOAD
from datasets.pdb import PDBSidechain
from datasets.pdbbind import NoiseTransform, PDBBind
from datasets.torsional import Torsional, TorsionalNoiseTransform
from bootstrapping.buffer import CBBuffer
from bootstrapping.bootstrapping import BootstrappingDataset
from utils.utils import read_strings_from_txt


class CombineDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        super(CombineDatasets, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def len(self):
        return len(self.dataset1) + len(self.dataset2)

    def get(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

    def add_complexes(self, new_complex_list):
        self.dataset1.add_complexes(new_complex_list)


def construct_loader(args, t_to_sigma, device):
    val_dataset2 = None
    if args.dataset == 'torsional':
        transform = TorsionalNoiseTransform(t_to_sigma=t_to_sigma, alpha=args.sampling_alpha, beta=args.sampling_beta,
                                       tor_alpha=args.tor_alpha, tor_beta=args.tor_beta,
                                       separate_noise_schedule=args.separate_noise_schedule,
                                       asyncronous_noise_schedule=args.asyncronous_noise_schedule)

        common_args = {'transform': transform, 'root': args.torsional_data_dir, 'limit_complexes': args.limit_complexes,
                       'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                       'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter, 'cache_path': args.cache_path,
                       'split_path': args.torsional_split_path, 'num_workers': args.num_workers, 'matching_tries': args.matching_tries}

        train_dataset = Torsional(mode='train', keep_original=True, **common_args)
        val_dataset = Torsional(mode='val', keep_original=True, **common_args)
        loader_class = DataLoader

    else:
        transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                                   all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                                   rot_alpha=args.rot_alpha, rot_beta=args.rot_beta, tor_alpha=args.tor_alpha,
                                   tor_beta=args.tor_beta, separate_noise_schedule=args.separate_noise_schedule,
                                   asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                                   include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                                   crop_beyond_cutoff=args.crop_beyond)
        if args.triple_training: assert args.combined_training

        sequences_to_embeddings = None
        if args.dataset == 'pdbsidechain' or args.triple_training:
            if args.pdbsidechain_esm_embeddings_path is not None:
                print('Loading ESM embeddings')
                id_to_embeddings = torch.load(args.pdbsidechain_esm_embeddings_path)
                sequences_list = read_strings_from_txt(args.pdbsidechain_esm_embeddings_sequences_path)
                sequences_to_embeddings = {}
                for i, seq in enumerate(sequences_list):
                    if str(i) in id_to_embeddings:
                        sequences_to_embeddings[seq] = id_to_embeddings[str(i)]

        if args.dataset == 'pdbsidechain' or args.triple_training:

            common_args = {'root': args.pdbsidechain_dir, 'transform': transform, 'limit_complexes': args.limit_complexes,
                           'receptor_radius': args.receptor_radius,
                           'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                           'remove_hs': args.remove_hs, 'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                           'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                           'knn_only_graph': not args.not_knn_only_graph, 'sequences_to_embeddings': sequences_to_embeddings,
                           'vandermers_max_dist': args.vandermers_max_dist,
                           'vandermers_buffer_residue_num': args.vandermers_buffer_residue_num,
                           'vandermers_min_contacts': args.vandermers_min_contacts,
                           'remove_second_segment': args.remove_second_segment,
                           'merge_clusters': args.merge_clusters}
            train_dataset3 = PDBSidechain(cache_path=args.cache_path, split='train', multiplicity=args.train_multiplicity, **common_args)

            if args.dataset == 'pdbsidechain':
                train_dataset = train_dataset3
                val_dataset = PDBSidechain(cache_path=args.cache_path, split='val', multiplicity=args.val_multiplicity, **common_args)
            loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

        if args.add_bootstrapping_dataset:
            transform2 = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                                       all_atom=args.all_atoms, alpha=args.bootstrap_alpha, beta=args.bootstrap_beta,
                                        minimum_t=args.bootstrap_tmin,
                                       rot_alpha=args.rot_alpha, rot_beta=args.rot_beta, tor_alpha=args.tor_alpha,
                                       tor_beta=args.tor_beta, separate_noise_schedule=args.separate_noise_schedule,
                                       asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                                       include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                                       crop_beyond_cutoff=args.crop_beyond)

            train_dataset4 = BootstrappingDataset(transform=transform2,
                                                  cache_path=args.cache_path,
                                                  original_model_dir=args.original_model_dir,
                                                  model_ckpt=args.original_model_ckpt,
                                                  confidence_model_dir=args.confidence_model_dir,
                                                  confidence_model_ckpt=args.confidence_model_ckpt,
                                                  device=device,
                                                  limit_complexes=args.limit_complexes,
                                                  sigma_schedule=args.sigma_schedule,
                                                  inference_steps=args.inference_steps,
                                                  inf_sched_alpha=args.inf_sched_alpha,
                                                  inf_sched_beta=args.inf_sched_beta,
                                                  rot_inf_sched_alpha=args.rot_inf_sched_alpha,
                                                  rot_inf_sched_beta=args.rot_inf_sched_beta,
                                                  tor_inf_sched_alpha=args.tor_inf_sched_alpha,
                                                  tor_inf_sched_beta=args.tor_inf_sched_beta,
                                                  samples_per_complex=args.samples_per_complex,
                                                  different_schedules=args.different_schedules,
                                                  all_atoms=args.all_atoms,
                                                  args=args,
                                                  multiplicity=args.train_multiplicity,
                                                  cache_ids_to_combine=args.cache_ids_to_combine,
                                                  cache_creation_id=args.cache_creation_id,
                                                  iterations=args.bootstrapping_dataset_iterations,
                                                  confidence_cutoff=args.bootstrapping_dataset_confidence_cutoff,
                                                  fixed_length=args.bootstrapping_fixed_length,
                                                  temperature=args.bootstrapping_temperature,
                                                  max_per_complex=args.bootstrapping_dataset_max_per_complex)

        if args.dataset in ['pdbbind', 'moad', 'generalisation', 'distillation']:
            common_args = {'transform': transform, 'limit_complexes': args.limit_complexes,
                           'chain_cutoff': args.chain_cutoff, 'receptor_radius': args.receptor_radius,
                           'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                           'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                           'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                           'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                           'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                           'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                           'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                           'matching_tries': args.matching_tries}

            if args.dataset == 'distillation': # TODO: Arthur - When is this ever used?
                train_dataset = CBBuffer(complexes_save_dir=args.distillation_complexes_dir,
                                                    cluster_name=args.distillation_train_cluster,
                                                    results_path=args.inference_out_dir, confidence_cutoff=args.confidence_cutoff,
                                                    transform=transform)

                val_dataset = MOAD(cache_path=args.cache_path, split='val', single_cluster_name=args.distillation_train_cluster,
                                   keep_original=True, multiplicity=args.val_multiplicity, max_receptor_size=args.max_receptor_size,
                                    remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                                    esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                                    unroll_clusters=args.unroll_clusters, root=args.moad_dir,
                                    esm_embeddings_path=args.moad_esm_embeddings_path, **common_args)

            if args.dataset == 'pdbbind' or args.dataset == 'generalisation' or args.combined_training:
                train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                                        num_conformers=args.num_conformers, root=args.pdbbind_dir,
                                        esm_embeddings_path=args.pdbbind_esm_embeddings_path,
                                        protein_file=args.protein_file, bootstrapping=args.add_bootstrapping_dataset, **common_args)

            if args.dataset == 'moad' or args.combined_training:
                train_dataset2 = MOAD(cache_path=args.cache_path, split='train', keep_original=True,
                                      num_conformers=args.num_conformers, max_receptor_size=args.max_receptor_size,
                                      remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                                      multiplicity= args.train_multiplicity, unroll_clusters=args.unroll_clusters,
                                      esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                                      root=args.moad_dir, esm_embeddings_path=args.moad_esm_embeddings_path,
                                      enforce_timesplit=args.enforce_timesplit, **common_args)

                if args.combined_training:
                    train_dataset = CombineDatasets(train_dataset2, train_dataset)
                    if args.triple_training:
                        train_dataset = CombineDatasets(train_dataset, train_dataset3)
                else:
                    train_dataset = train_dataset2

            if args.add_bootstrapping_dataset:
                len_before = len(train_dataset)
                train_dataset = CombineDatasets(train_dataset, train_dataset4)
                print(f'With bootstrapping training dataset size went from {len_before} to {len(train_dataset)}')

            if args.dataset == 'pdbbind' or args.double_val:
                val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True,
                                      esm_embeddings_path=args.pdbbind_esm_embeddings_path, root=args.pdbbind_dir,
                                      protein_file=args.protein_file, require_ligand=True, **common_args)
                if args.double_val:
                    val_dataset2 = val_dataset

            if args.dataset == 'moad' or args.dataset == 'generalisation':
                val_dataset = MOAD(cache_path=args.cache_path, split='val', keep_original=True,
                                   multiplicity= args.val_multiplicity, max_receptor_size=args.max_receptor_size,
                                   remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                                   esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                                   unroll_clusters=args.unroll_clusters, root=args.moad_dir,
                                   esm_embeddings_path=args.moad_esm_embeddings_path, require_ligand=True, **common_args)

            loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=False, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    return train_loader, val_loader, val_dataset2

