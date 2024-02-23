import copy
import os
import torch
import esm.data
from datasets.moad import MOAD
from utils.gnina_utils import get_gnina_poses
from utils.molecules_utils import get_symmetry_rmsd

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import scipy
import wandb
from biopandas.pdb import PandasPdb
import plotly.express as px
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveHs, RemoveAllHs

import subprocess
from datasets.process_mols import write_mol_with_coords, read_molecule
import re

from utils import so3

from datasets.pdbbind import PDBBind, read_mol, NoiseTransform
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, \
    get_inverse_schedule
from utils.diffusion_utils import set_time
from utils.sampling import randomize_position, sampling
from utils.training import loss_function
from utils.utils import get_model, remove_all_hs, read_strings_from_txt, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm
from relax.xtb import optimize_complex

RDLogger.DisableLog('rdApp.*')
import yaml
import pickle


def get_dataset(args, model_args, filtering=False):
    if args.dataset != 'moad':
        dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes, dataset=args.dataset,
                        chain_cutoff=args.chain_cutoff,
                        receptor_radius=model_args.receptor_radius,
                        cache_path=args.cache_path, split_path=args.split_path,
                        remove_hs=model_args.remove_hs, max_lig_size=None,
                        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
                        matching=not model_args.no_torsion, keep_original=True,
                        popsize=args.matching_popsize,
                        maxiter=args.matching_maxiter,
                        all_atoms=model_args.all_atoms if 'all_atoms' in model_args else False,
                        atom_radius=model_args.atom_radius if 'all_atoms' in model_args else None,
                        atom_max_neighbors=model_args.atom_max_neighbors if 'all_atoms' in model_args else None,
                        esm_embeddings_path=args.esm_embeddings_path,
                        require_ligand=True,
                        num_workers=args.num_workers,
                        protein_file=args.protein_file,
                        ligand_file=args.ligand_file,
                        knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                        include_miscellaneous_atoms=False if not hasattr(args,
                                                                         'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                        num_conformers=args.samples_per_complex if args.resample_rdkit and not filtering else 1)

    else:
        dataset = MOAD(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                       chain_cutoff=args.chain_cutoff,
                       receptor_radius=model_args.receptor_radius,
                       cache_path=args.cache_path, split=args.split,
                       remove_hs=model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
                       matching=not model_args.no_torsion, keep_original=True,
                       popsize=args.matching_popsize,
                       maxiter=args.matching_maxiter,
                       all_atoms=model_args.all_atoms if 'all_atoms' in model_args else False,
                       atom_radius=model_args.atom_radius if 'all_atoms' in model_args else None,
                       atom_max_neighbors=model_args.atom_max_neighbors if 'all_atoms' in model_args else None,
                       esm_embeddings_path=args.esm_embeddings_path,
                       esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                       require_ligand=True,
                       num_workers=args.num_workers,
                       knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                       include_miscellaneous_atoms=False if not hasattr(args,
                                                                        'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                       num_conformers=args.samples_per_complex if args.resample_rdkit and not filtering else 1,
                       unroll_clusters=args.unroll_clusters, remove_pdbbind=args.remove_pdbbind,
                       min_ligand_size=args.min_ligand_size,
                       max_receptor_size=args.max_receptor_size,
                       single_cluster_name=args.single_cluster_name,
                       remove_promiscuous_targets=args.remove_promiscuous_targets,
                       no_randomness=True,
                       skip_matching=args.skip_matching)
    return dataset



if __name__ == '__main__':
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--model_dir', type=str, default='workdir', help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--filtering_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--filtering_ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
    parser.add_argument('--run_name', type=str, default='test', help='')
    parser.add_argument('--project', type=str, default='ligbind_inf', help='')
    parser.add_argument('--out_dir', type=str, default=None, help='Where to save results to')
    parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')

    parser.add_argument('--old_score_model', action='store_true', default=False, help='')
    parser.add_argument('--old_filtering_model', action='store_true', default=False, help='')
    parser.add_argument('--matching_popsize', type=int, default=40, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=40, help='Differential evolution maxiter parameter in matching')

    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')
    parser.add_argument('--moad_esm_embeddings_sequences_path', type=str, default=None, help='')
    parser.add_argument('--chain_cutoff', type=float, default=None, help='Cutoff of the chains from the ligand') # TODO remove
    parser.add_argument('--use_full_size_protein_file', action='store_true', default=False, help='') # TODO remove
    parser.add_argument('--use_original_protein_file', action='store_true', default=False, help='') # TODO remove
    parser.add_argument('--save_complexes', action='store_true', default=False, help='Save generated complex graphs')
    parser.add_argument('--complexes_save_path', type=str, default=None, help='')

    parser.add_argument('--dataset', type=str, default='moad', help='')
    parser.add_argument('--cache_path', type=str, default='data/cacheMOAD', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/', help='Folder containing original structures')
    parser.add_argument('--split_path', type=str, default='data/BindingMOAD_2020_ab_processed/splits/val.txt', help='Path of file defining the split')

    parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
    parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
    parser.add_argument('--overwrite_no_final_step_noise', default=False, help='This sets wandb to True if it is True. It exists for wandb sweeps.') # TODO remove
    parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
    parser.add_argument('--wandb', action='store_true', default=False, help='') # TODO remove
    parser.add_argument('--overwrite_wandb', default=False, help='This sets wandb to True if it is True. It exists for wandb sweeps.') # TODO remove
    parser.add_argument('--inference_steps', type=int, default=5, help='Number of denoising steps')
    parser.add_argument('--limit_complexes', type=int, default=10, help='Limit to the number of complexes')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
    parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
    parser.add_argument('--save_visualisation', action='store_true', default=True, help='Whether to save visualizations')
    parser.add_argument('--samples_per_complex', type=int, default=4, help='Number of poses to sample for each complex')
    parser.add_argument('--resample_rdkit', action='store_true', default=False, help='')
    parser.add_argument('--skip_matching', action='store_true', default=False, help='')
    parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='Schedule type, no other options')
    parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--different_schedules', action='store_true', default=False, help='')
    parser.add_argument('--overwrite_different_schedules', default=False, help='This sets different_schedules to True if it is True. It exists for wandb sweeps.')
    parser.add_argument('--rot_sigma_schedule', type=str, default='expbeta', help='')
    parser.add_argument('--rot_inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--rot_inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--tor_sigma_schedule', type=str, default='expbeta', help='')
    parser.add_argument('--tor_inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--tor_inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--pocket_knowledge', action='store_true', default=False, help='')
    parser.add_argument('--no_random_pocket', action='store_true', default=False, help='')
    parser.add_argument('--overwrite_pocket_knowledge', default=False, help='')
    parser.add_argument('--pocket_tr_max', type=float, default=3, help='')
    parser.add_argument('--pocket_cutoff', type=float, default=5, help='')
    parser.add_argument('--actual_steps', type=int, default=None, help='')
    parser.add_argument('--xtb', action='store_true', default=False, help='')
    parser.add_argument('--use_true_pivot', action='store_true', default=False, help='')
    parser.add_argument('--restrict_cpu', action='store_true', default=False, help='')
    parser.add_argument('--force_fixed_center_conv', action='store_true', default=False, help='')
    parser.add_argument('--protein_file', type=str, default='protein_processed', help='')
    parser.add_argument('--unroll_clusters', action='store_true', default=True, help='')
    parser.add_argument('--ligand_file', type=str, default='ligand', help='')
    parser.add_argument('--remove_pdbbind', action='store_true', default=False, help='')
    parser.add_argument('--split', type=str, default='val', help='')
    parser.add_argument('--limit_failures', type=float, default=5, help='')
    parser.add_argument('--min_ligand_size', type=float, default=0, help='')
    parser.add_argument('--max_receptor_size', type=float, default=None, help='')
    parser.add_argument('--remove_promiscuous_targets', type=float, default=None, help='')

    parser.add_argument('--svgd_weight_log_0', type=float, default=None)
    parser.add_argument('--svgd_weight_log_1', type=float, default=None)
    parser.add_argument('--svgd_repulsive_weight_log_0', type=float, default=None)
    parser.add_argument('--svgd_repulsive_weight_log_1', type=float, default=None)
    parser.add_argument('--svgd_langevin_weight_log_0', type=float, default=None)
    parser.add_argument('--svgd_langevin_weight_log_1', type=float, default=None)
    parser.add_argument('--svgd_kernel_size_log_0', type=float, default=None)
    parser.add_argument('--svgd_kernel_size_log_1', type=float, default=None)
    parser.add_argument('--svgd_rot_log_rel_weight', type=float, default=0.0)
    parser.add_argument('--svgd_tor_log_rel_weight', type=float, default=0.0)
    parser.add_argument('--svgd_weight_log', type=float, default=None)
    parser.add_argument('--svgd_repulsive_weight_log', type=float, default=None)
    parser.add_argument('--svgd_use_x0', type=bool, default=False)

    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data', type=float, default=0.5)
    parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
    parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='') # To redirect gnina subprocesses stdouts from the terminal window
    parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
    parser.add_argument('--save_gnina_metrics', action='store_true', default=False, help='')
    parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
    parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)
    parser.add_argument('--single_cluster_name', type=str, default=None, help='')
    
    args = parser.parse_args()

    if args.svgd_weight_log is not None:
        args.svgd_weight = 10**args.svgd_weight_log
    if args.svgd_repulsive_weight_log is not None:
        args.svgd_repulsive_weight = 10**args.svgd_repulsive_weight_log

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    if args.restrict_cpu:
        threads = 16
        os.environ["OMP_NUM_THREADS"] = str(threads)  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = str(threads)  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)  # export NUMEXPR_NUM_THREADS=6
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        torch.set_num_threads(threads)
    if args.overwrite_different_schedules:  # This sets different_schedules to True if it is True. It exists for wandb sweeps.
        args.different_schedules = True
    if args.overwrite_pocket_knowledge:  # This sets different_schedules to True if it is True. It exists for wandb sweeps.
        args.pocket_knowledge = True
    if args.overwrite_wandb:  # This sets wandb to True if it is True. It exists for wandb sweeps.
        args.wandb = True
    if args.overwrite_no_final_step_noise:  # This sets wandb to True if it is True. It exists for wandb sweeps.
        args.no_final_step_noise = True

    if args.out_dir is None: args.out_dir = f'inference_out_dir_not_specified/{args.run_name}'
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
        if not hasattr(score_model_args, 'separate_noise_schedule'):  # exists for compatibility with old runs that did not have the attribute
            score_model_args.separate_noise_schedule = False
        if not hasattr(score_model_args, 'lm_embeddings_path'):
            score_model_args.lm_embeddings_path = None
        if not hasattr(score_model_args, 'tr_only_confidence'):
            score_model_args.tr_only_confidence = True
        if not hasattr(score_model_args, 'high_confidence_threshold'):
            score_model_args.high_confidence_threshold = 0.0
        if not hasattr(score_model_args, 'include_confidence_prediction'):
            score_model_args.include_confidence_prediction = False
        if not hasattr(score_model_args, 'confidence_weight'):
            score_model_args.confidence_weight = 1
        if not hasattr(score_model_args, 'asyncronous_noise_schedule'):
            score_model_args.asyncronous_noise_schedule = False
        if not hasattr(score_model_args, 'correct_torsion_sigmas'):
            score_model_args.correct_torsion_sigmas = False
        if not hasattr(score_model_args, 'esm_embeddings_path'):
            score_model_args.esm_embeddings_path = None
        if args.force_fixed_center_conv:
            score_model_args.not_fixed_center_conv = False
    if args.filtering_model_dir is not None:
        with open(f'{args.filtering_model_dir}/model_parameters.yml') as f:
            filtering_args = Namespace(**yaml.full_load(f))
        if not os.path.exists(filtering_args.original_model_dir):
            print("Path does not exist: ", filtering_args.original_model_dir)
            filtering_args.original_model_dir = os.path.join(*filtering_args.original_model_dir.split('/')[-2:])
            print('instead trying path: ', filtering_args.original_model_dir)
        if not hasattr(filtering_args, 'use_original_model_cache'):
            filtering_args.use_original_model_cache = True
        if not hasattr(filtering_args, 'esm_embeddings_path'):
            filtering_args.esm_embeddings_path = None
        if not hasattr(filtering_args, 'num_classification_bins'):
            filtering_args.num_classification_bins = 2

    if args.num_cpu is not None:
        torch.set_num_threads(args.num_cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = get_dataset(args, score_model_args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    if args.filtering_model_dir is not None:
        if not (filtering_args.use_original_model_cache or filtering_args.transfer_weights):
            # if the filtering model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            print('HAPPENING | filtering model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the filtering model now.')
            filtering_test_dataset = get_dataset(args, filtering_args, filtering=True)
            filtering_complex_dict = {d.name: d for d in filtering_test_dataset}

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    if not args.no_model:
        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
        if args.ckpt == 'last_model.pt':
            model_state_dict = state_dict['model']
            ema_weights_state = state_dict['ema_weights']
            model.load_state_dict(model_state_dict, strict=True)
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
            ema_weights.load_state_dict(ema_weights_state, device=device)
            ema_weights.copy_to(model.parameters())
        else:
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            model.eval()
        if args.filtering_model_dir is not None:
            if filtering_args.transfer_weights:
                with open(f'{filtering_args.original_model_dir}/model_parameters.yml') as f:
                    filtering_model_args = Namespace(**yaml.full_load(f))
                if not hasattr(filtering_model_args, 'separate_noise_schedule'):  # exists for compatibility with old runs that did not have the
                    # attribute
                    filtering_model_args.separate_noise_schedule = False
                if not hasattr(filtering_model_args, 'lm_embeddings_path'):
                    filtering_model_args.lm_embeddings_path = None
                if not hasattr(filtering_model_args, 'tr_only_confidence'):
                    filtering_model_args.tr_only_confidence = True
                if not hasattr(filtering_model_args, 'high_confidence_threshold'):
                    filtering_model_args.high_confidence_threshold = 0.0
                if not hasattr(filtering_model_args, 'include_confidence_prediction'):
                    filtering_model_args.include_confidence_prediction = False
                if not hasattr(filtering_model_args, 'confidence_dropout'):
                    filtering_model_args.confidence_dropout = filtering_model_args.dropout
                if not hasattr(filtering_model_args, 'confidence_no_batchnorm'):
                    filtering_model_args.confidence_no_batchnorm = False
                if not hasattr(filtering_model_args, 'confidence_weight'):
                    filtering_model_args.confidence_weight = 1
                if not hasattr(filtering_model_args, 'asyncronous_noise_schedule'):
                    filtering_model_args.asyncronous_noise_schedule = False
                if not hasattr(filtering_model_args, 'correct_torsion_sigmas'):
                    filtering_model_args.correct_torsion_sigmas = False
                if not hasattr(filtering_model_args, 'esm_embeddings_path'):
                    filtering_model_args.esm_embeddings_path = None
                if not hasattr(filtering_model_args, 'not_fixed_knn_radius_graph'):
                    filtering_model_args.not_fixed_knn_radius_graph = True
                if not hasattr(filtering_model_args, 'not_knn_only_graph'):
                    filtering_model_args.not_knn_only_graph = True
            else:
                filtering_model_args = filtering_args

            filtering_model = get_model(filtering_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                        confidence_mode=True, old=args.old_filtering_model)
            state_dict = torch.load(f'{args.filtering_model_dir}/{args.filtering_ckpt}', map_location=torch.device('cpu'))
            filtering_model.load_state_dict(state_dict, strict=True)
            filtering_model = filtering_model.to(device)
            filtering_model.eval()
        else:
            filtering_model = None
            filtering_args = None
            filtering_model_args = None

    if args.wandb:
        run = wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )

    if args.pocket_knowledge and args.different_schedules:
        t_max = (np.log(args.pocket_tr_max) - np.log(score_model_args.tr_sigma_min)) / (
                    np.log(score_model_args.tr_sigma_max) - np.log(score_model_args.tr_sigma_min))
    else:
        t_max = 1

    tr_schedule = get_t_schedule(sigma_schedule=args.sigma_schedule, inference_steps=args.inference_steps,
                                 inf_sched_alpha=args.inf_sched_alpha, inf_sched_beta=args.inf_sched_beta,
                                 t_max=t_max)

    t_schedule = None  # used only by asyncronous_noise_schedule
    if args.different_schedules:
        rot_schedule = get_t_schedule(sigma_schedule=args.rot_sigma_schedule, inference_steps=args.inference_steps,
                                      inf_sched_alpha=args.rot_inf_sched_alpha, inf_sched_beta=args.rot_inf_sched_beta)
        tor_schedule = get_t_schedule(sigma_schedule=args.tor_sigma_schedule, inference_steps=args.inference_steps,
                                      inf_sched_alpha=args.tor_inf_sched_alpha, inf_sched_beta=args.tor_inf_sched_beta)
        print('tr schedule', tr_schedule)
        print('rot schedule', rot_schedule)
        print('tor schedule', tor_schedule)

    elif score_model_args.asyncronous_noise_schedule:
        print("asyncronous_noise_schedule")
        t_schedule = tr_schedule
        tr_schedule = get_inverse_schedule(t_schedule, score_model_args.sampling_alpha, score_model_args.sampling_beta)
        rot_schedule = get_inverse_schedule(t_schedule, score_model_args.rot_alpha, score_model_args.rot_beta)
        tor_schedule = get_inverse_schedule(t_schedule, score_model_args.tor_alpha, score_model_args.tor_beta)
        print('tr schedule', tr_schedule)
        print('rot schedule', rot_schedule)
        print('tor schedule', tor_schedule)
    else:
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print('common t schedule', tr_schedule)

    rmsds_list, obrmsds, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list, names_list = [], [], [], 0, 0, [], [], [], []
    true_affinities_list, pred_affinities_list, run_times, min_self_distances_list, without_rec_overlap_list = [], [], [], [], []
    gnina_rmsds_list, gnina_score_list = [], []
    N = args.samples_per_complex
    #names_no_rec_overlap = read_strings_from_txt(f'data/splits/timesplit_test_no_rec_overlap')
    names_no_rec_overlap = np.load("data/BindingMOAD_2020_ab_processed_biounit/test_names_bootstrapping.npy")
    print('Size of test dataset: ', len(test_dataset))

    #limit_test = ['3zlw']
    if args.save_complexes:
        sampled_complexes = {}

    if args.save_gnina_metrics:
        # key is complex_name, value is the gnina metrics for all samples
        gnina_metrics = {}

    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        torch.cuda.empty_cache()

        if filtering_model is not None and not (
                filtering_args.use_original_model_cache or filtering_args.transfer_weights) and orig_complex_graph.name[
            0] not in filtering_complex_dict.keys():
            skipped += 1
            print(
                f"HAPPENING | The filtering dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
            continue

        success = 0
        bs = args.batch_size
        while 0 >= success > -args.limit_failures:
            try:
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                if args.resample_rdkit:
                    for i, g in enumerate(data_list):
                        g['ligand'].pos = g['ligand'].pos[i]

                pivot = None
                if args.use_true_pivot:
                    pivot = orig_complex_graph['ligand'].pos
                randomize_position(data_list, score_model_args.no_torsion, args.no_random or args.no_random_pocket,
                                   score_model_args.tr_sigma_max if not args.pocket_knowledge else args.pocket_tr_max,
                                   args.pocket_knowledge, args.pocket_cutoff)


                pdb = None
                if args.save_visualisation:
                    visualization_list = []
                    for idx, graph in enumerate(data_list):
                        lig = orig_complex_graph.mol[0]
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add(((orig_complex_graph['ligand'].pos if not args.resample_rdkit else orig_complex_graph['ligand'].pos[idx]) + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                start_time = time.time()
                if not args.no_model:
                    if filtering_model is not None and not (
                            filtering_args.use_original_model_cache or filtering_args.transfer_weights):
                        filtering_data_list = [copy.deepcopy(filtering_complex_dict[orig_complex_graph.name[0]]) for _ in
                                               range(N)]
                    else:
                        filtering_data_list = None

                    data_list, confidence = sampling(data_list=data_list, model=model,
                                                     inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                     tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                     tor_schedule=tor_schedule,
                                                     device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                     no_random=args.no_random,
                                                     ode=args.ode, visualization_list=visualization_list,
                                                     confidence_model=filtering_model,
                                                     filtering_data_list=filtering_data_list,
                                                     filtering_model_args=filtering_model_args,
                                                     asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule,
                                                     t_schedule=t_schedule,
                                                     batch_size=bs,
                                                     no_final_step_noise=args.no_final_step_noise, pivot=pivot,
                                                     temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                                                     temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                                     temp_sigma_data=args.temp_sigma_data,
                                                     svgd_weight_log_0=args.svgd_weight_log_0,
                                                     svgd_weight_log_1=args.svgd_weight_log_1,
                                                     svgd_repulsive_weight_log_0=args.svgd_repulsive_weight_log_0,
                                                     svgd_repulsive_weight_log_1=args.svgd_repulsive_weight_log_1,
                                                     svgd_kernel_size_log_0=args.svgd_kernel_size_log_0,
                                                     svgd_kernel_size_log_1=args.svgd_kernel_size_log_1,
                                                     svgd_langevin_weight_log_0=args.svgd_langevin_weight_log_0,
                                                     svgd_langevin_weight_log_1=args.svgd_langevin_weight_log_1,
                                                     svgd_rot_log_rel_weight=args.svgd_rot_log_rel_weight,
                                                     svgd_tor_log_rel_weight=args.svgd_tor_log_rel_weight,
                                                     svgd_use_x0=args.svgd_use_x0)

                    
                    if args.xtb:
                        print(len(data_list), confidence[:, 0].shape)
                        conf = confidence[:, 0].cpu().numpy()
                        idx = np.argmax(conf)
                        print(idx)
                        optimize_complex(data_list[idx])
                run_times.append(time.time() - start_time)
                if score_model_args.no_torsion:
                    orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

                filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    # Same pair with multiple binding positions
                    # print(f'Number of ground truth poses: {len(orig_complex_graph['ligand'].orig_pos)}')
                    if args.dataset == 'moad':
                        orig_ligand_pos = np.array([pos[filterHs] - orig_complex_graph.original_center.cpu().numpy() for pos in orig_complex_graph['ligand'].orig_pos[0]])
                    else:
                        orig_ligand_pos = np.array([pos[filterHs] - orig_complex_graph.original_center.cpu().numpy() for pos in [orig_complex_graph['ligand'].orig_pos[0]]])
                else:
                    print('default path')
                    orig_ligand_pos = np.expand_dims(
                        orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                        axis=0)

                ligand_pos = np.asarray(
                        [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])

                # Use gnina to minimize energy for predicted ligands.
                if args.gnina_minimize:
                    print('Running gnina on all predicted ligand positions for energy minimization.')
                    gnina_rmsds, gnina_scores = [], []
                    lig = copy.deepcopy(orig_complex_graph.mol[0])
                    positions = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list])

                    conf = confidence
                    if conf is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
                        conf = conf[:, 0]
                    if conf is not None:
                        conf = conf.cpu().numpy()
                        conf = np.nan_to_num(conf, nan=-1e-6)
                        re_order = np.argsort(conf)[::-1]
                        positions = positions[re_order]

                    for pos in positions[:args.gnina_poses_to_optimize]:
                        gnina_ligand_pos, gnina_mol, gnina_score = get_gnina_poses(args, lig, pos, orig_complex_graph.original_center.cpu().numpy(), orig_complex_graph.name[0])

                        mol = RemoveAllHs(orig_complex_graph.mol[0])
                        rmsds = []
                        for i in range(len(orig_ligand_pos)):
                            try:
                                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], gnina_ligand_pos, gnina_mol)
                            except Exception as e:
                                print("Using non corrected RMSD because of the error:", e)
                                rmsd = np.sqrt(((gnina_ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=1).mean(axis=0))
                            rmsds.append(rmsd)
                        rmsds = np.asarray(rmsds)
                        rmsd = np.min(rmsds, axis=0)
                        gnina_rmsds.append(rmsd)
                        gnina_scores.append(gnina_score)

                    gnina_rmsds = np.asarray(gnina_rmsds)
                    assert gnina_rmsds.shape == (args.gnina_poses_to_optimize,), str(gnina_rmsds.shape) + " " + str(args.gnina_poses_to_optimize)
                    gnina_rmsds_list.append(gnina_rmsds)
                    gnina_scores = np.asarray(gnina_scores)
                    gnina_score_list.append(gnina_scores)

                mol = RemoveAllHs(orig_complex_graph.mol[0])
                rmsds = []
                for i in range(len(orig_ligand_pos)):
                    try:
                        rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
                    except Exception as e:
                        print("Using non corrected RMSD because of the error:", e)
                        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
                    rmsds.append(rmsd)
                rmsds = np.asarray(rmsds)
                rmsd = np.min(rmsds, axis=0)
                
                centroid_distance = np.min(np.linalg.norm(ligand_pos.mean(axis=1)[None, :] - orig_ligand_pos.mean(axis=1)[:, None], axis=2), axis=0)

                if confidence is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    confidence = np.nan_to_num(confidence, nan=-1e-6)
                    re_order = np.argsort(confidence)[::-1]
                    print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1)[re_order], ' centroid distance',
                          np.around(centroid_distance, 1)[re_order], ' confidences ', np.around(confidence, 4)[re_order],
                          (' gnina rmsd ' + str(np.around(gnina_rmsds, 1))) if args.gnina_minimize else '')
                    confidences_list.append(confidence)
                else:
                    print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1), ' centroid distance',
                          np.around(centroid_distance, 1))
                centroid_distances_list.append(centroid_distance)

                self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

                if args.save_complexes:
                    sampled_complexes[orig_complex_graph.name[0]] = data_list

                if args.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(
                                f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}.pdb')
                    else:
                        for rank, batch_idx in enumerate(np.argsort(rmsd)):
                            visualization_list[batch_idx].write(
                                f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
                without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
                names_list.append(orig_complex_graph.name[0])
                rmsds_list.append(rmsd)
                success = 1
            except Exception as e:
                print("Failed on", orig_complex_graph["name"], e)
                success -= 1
                if bs > 1:
                    bs = bs // 2

        if success != 1:
            rmsds_list.append(np.zeros(args.samples_per_complex) + 10000)
            if filtering_model_args is not None:
                confidences_list.append(np.zeros(args.samples_per_complex) - 10000)
            centroid_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
            min_self_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
            without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
            names_list.append(orig_complex_graph.name[0])
            failures += 1

    print('Performance without hydrogens included in the loss')
    print(failures, "failures due to exceptions")
    print(skipped, ' skipped because complex was not in filtering dataset')

    if args.save_complexes:
        print("Saving complexes.")
        if args.complexes_save_path is not None:
            with open(os.path.join(args.complexes_save_path, "ligands.pkl"), 'wb') as f:
                pickle.dump(sampled_complexes, f)
    
    if args.save_gnina_metrics:
        with open(f'{args.out_dir}/gnina_metrics.pkl', 'wb') as f:
            pickle.dump(gnina_metrics, f)
        print("Saved gnina metrics")

    performance_metrics = {}
    for overlap in ['', 'no_overlap_']:
        if 'no_overlap_' == overlap:
            without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
            if without_rec_overlap.sum() == 0: continue
            rmsds = np.array(rmsds_list)[without_rec_overlap]
            min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
            centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
            if args.filtering_model_dir is not None:
                confidences = np.array(confidences_list)[without_rec_overlap]
            else:
                confidences = np.array(confidences_list)
            names = np.array(names_list)[without_rec_overlap]
            gnina_rmsds = np.array(gnina_rmsds_list)[without_rec_overlap] if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list)[without_rec_overlap] if args.gnina_minimize else None

        else:
            rmsds = np.array(rmsds_list)
            gnina_rmsds = np.array(gnina_rmsds_list) if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list) if args.gnina_minimize else None
            min_self_distances = np.array(min_self_distances_list)
            centroid_distances = np.array(centroid_distances_list)
            confidences = np.array(confidences_list)
            names = np.array(names_list)

        run_times = np.array(run_times)
        np.save(f'{args.out_dir}/{overlap}min_self_distances.npy', min_self_distances)
        np.save(f'{args.out_dir}/{overlap}rmsds.npy', rmsds)
        np.save(f'{args.out_dir}/{overlap}centroid_distances.npy', centroid_distances)
        np.save(f'{args.out_dir}/{overlap}confidences.npy', confidences)
        np.save(f'{args.out_dir}/{overlap}run_times.npy', run_times)
        np.save(f'{args.out_dir}/{overlap}complex_names.npy', np.array(names))
        np.save(f'{args.out_dir}/{overlap}gnina_rmsds.npy', gnina_rmsds)
        np.save(f'{args.out_dir}/{overlap}gnina_score.npy', gnina_score)

        print(rmsds)
        print(gnina_rmsds)

        performance_metrics.update({
            f'{overlap}run_times_std': run_times.std().__round__(2),
            f'{overlap}run_times_mean': run_times.mean().__round__(2),
            f'{overlap}mean_rmsd': rmsds.mean(),
            f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
            f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N),
            f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
            f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
            f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),
            f'{overlap}min_rmsds_below_2': (100 * (np.min(rmsds, axis=1) < 2).sum() / len(rmsds)),
            f'{overlap}min_rmsds_below_5': (100 * (np.min(rmsds, axis=1) < 5).sum() / len(rmsds)),

            f'{overlap}mean_centroid': centroid_distances.mean().__round__(2),
            f'{overlap}centroid_below_2': (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
            f'{overlap}centroid_below_5': (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),
            f'{overlap}centroid_percentile_25': np.percentile(centroid_distances, 25).round(2),
            f'{overlap}centroid_percentile_50': np.percentile(centroid_distances, 50).round(2),
            f'{overlap}centroid_percentile_75': np.percentile(centroid_distances, 75).round(2),
        })

        if args.gnina_minimize:
            score_ordering = np.argsort(gnina_score, axis=1)[:, ::-1]
            filtered_rmsds_gnina = gnina_rmsds[np.arange(gnina_rmsds.shape[0])[:, None], score_ordering][:, 0]

            performance_metrics.update({
                f'{overlap}gnina_rmsds_below_2': (100 * (gnina_rmsds < 2).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize) if args.gnina_minimize else None,
                f'{overlap}gnina_rmsds_below_5': (100 * (gnina_rmsds < 5).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize) if args.gnina_minimize else None,
                f'{overlap}gnina_min_rmsds_below_2': (100 * (np.min(gnina_rmsds, axis=1) < 2).sum() / len(gnina_rmsds)) if args.gnina_minimize else None,
                f'{overlap}gnina_min_rmsds_below_5': (100 * (np.min(gnina_rmsds, axis=1) < 5).sum() / len(gnina_rmsds)) if args.gnina_minimize else None,
                f'{overlap}gnina_filtered_rmsds_below_2': (100 * (filtered_rmsds_gnina < 2).sum() / len(filtered_rmsds_gnina)).__round__(2),
                f'{overlap}gnina_filtered_rmsds_below_5': (100 * (filtered_rmsds_gnina < 5).sum() / len(filtered_rmsds_gnina)).__round__(2),
                f'{overlap}gnina_rmsds_percentile_25': np.percentile(gnina_rmsds, 25).round(2),
                f'{overlap}gnina_rmsds_percentile_50': np.percentile(gnina_rmsds, 50).round(2),
                f'{overlap}gnina_rmsds_percentile_75': np.percentile(gnina_rmsds, 75).round(2),

            })

        if N >= 5:
            top5_rmsds = np.min(rmsds[:, :5], axis=1)
            top5_centroid_distances = centroid_distances[
                                          np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
            top5_min_self_distances = min_self_distances[
                                          np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
            performance_metrics.update({
                f'{overlap}top5_self_intersect_fraction': (
                            100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),
                f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
                f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
                f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
                f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
                f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

                f'{overlap}top5_centroid_below_2': (
                            100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
                f'{overlap}top5_centroid_below_5': (
                            100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
                f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
                f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
                f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
            })

        if N >= 10:
            top10_rmsds = np.min(rmsds[:, :10], axis=1)
            top10_centroid_distances = centroid_distances[
                                           np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
            top10_min_self_distances = min_self_distances[
                                           np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
            performance_metrics.update({
                f'{overlap}top10_self_intersect_fraction': (
                            100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),
                f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
                f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
                f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
                f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
                f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

                f'{overlap}top10_centroid_below_2': (
                            100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
                f'{overlap}top10_centroid_below_5': (
                            100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
                f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
                f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
                f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
            })

        if filtering_model is not None:
            confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]

            filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
            filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
            filtered_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
            performance_metrics.update({
                f'{overlap}filtered_self_intersect_fraction': (
                            100 * (filtered_min_self_distances < 0.4).sum() / len(filtered_min_self_distances)).__round__(
                    2),
                f'{overlap}filtered_rmsds_below_2': (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),
                f'{overlap}filtered_rmsds_below_5': (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
                f'{overlap}filtered_rmsds_percentile_25': np.percentile(filtered_rmsds, 25).round(2),
                f'{overlap}filtered_rmsds_percentile_50': np.percentile(filtered_rmsds, 50).round(2),
                f'{overlap}filtered_rmsds_percentile_75': np.percentile(filtered_rmsds, 75).round(2),

                f'{overlap}filtered_centroid_below_2': (
                            100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)).__round__(2),
                f'{overlap}filtered_centroid_below_5': (
                            100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)).__round__(2),
                f'{overlap}filtered_centroid_percentile_25': np.percentile(filtered_centroid_distances, 25).round(2),
                f'{overlap}filtered_centroid_percentile_50': np.percentile(filtered_centroid_distances, 50).round(2),
                f'{overlap}filtered_centroid_percentile_75': np.percentile(filtered_centroid_distances, 75).round(2),
            })

            if N >= 5:
                top5_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)
                top5_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                top5_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top5_filtered_rmsds_below_2': (
                                100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)).__round__(2),
                    f'{overlap}top5_filtered_rmsds_below_5': (
                                100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)).__round__(2),
                    f'{overlap}top5_filtered_rmsds_percentile_25': np.percentile(top5_filtered_rmsds, 25).round(2),
                    f'{overlap}top5_filtered_rmsds_percentile_50': np.percentile(top5_filtered_rmsds, 50).round(2),
                    f'{overlap}top5_filtered_rmsds_percentile_75': np.percentile(top5_filtered_rmsds, 75).round(2),

                    f'{overlap}top5_filtered_centroid_below_2': (100 * (top5_filtered_centroid_distances < 2).sum() / len(
                        top5_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top5_filtered_centroid_below_5': (100 * (top5_filtered_centroid_distances < 5).sum() / len(
                        top5_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top5_filtered_centroid_percentile_25': np.percentile(top5_filtered_centroid_distances,
                                                                                    25).round(2),
                    f'{overlap}top5_filtered_centroid_percentile_50': np.percentile(top5_filtered_centroid_distances,
                                                                                    50).round(2),
                    f'{overlap}top5_filtered_centroid_percentile_75': np.percentile(top5_filtered_centroid_distances,
                                                                                    75).round(2),
                })
            if N >= 10:
                top10_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
                                              axis=1)
                top10_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                top10_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top10_filtered_rmsds_below_2': (
                                100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)).__round__(2),
                    f'{overlap}top10_filtered_rmsds_below_5': (
                                100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)).__round__(2),
                    f'{overlap}top10_filtered_rmsds_percentile_25': np.percentile(top10_filtered_rmsds, 25).round(2),
                    f'{overlap}top10_filtered_rmsds_percentile_50': np.percentile(top10_filtered_rmsds, 50).round(2),
                    f'{overlap}top10_filtered_rmsds_percentile_75': np.percentile(top10_filtered_rmsds, 75).round(2),

                    f'{overlap}top10_filtered_centroid_below_2': (100 * (top10_filtered_centroid_distances < 2).sum() / len(
                        top10_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top10_filtered_centroid_below_5': (100 * (top10_filtered_centroid_distances < 5).sum() / len(
                        top10_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top10_filtered_centroid_percentile_25': np.percentile(top10_filtered_centroid_distances,
                                                                                     25).round(2),
                    f'{overlap}top10_filtered_centroid_percentile_50': np.percentile(top10_filtered_centroid_distances,
                                                                                     50).round(2),
                    f'{overlap}top10_filtered_centroid_percentile_75': np.percentile(top10_filtered_centroid_distances,
                                                                                     75).round(2),
                })

            reverse_confidence_ordering = np.argsort(confidences, axis=1)
            reverse_filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], reverse_confidence_ordering][:, 0]
            reverse_filtered_centroid_distances = centroid_distances[
                                                      np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
            reverse_filtered_min_self_distances = min_self_distances[
                                                      np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
            performance_metrics.update({
                f'{overlap}reversefiltered_self_intersect_fraction': (
                            100 * (reverse_filtered_min_self_distances < 0.4).sum() / len(
                        reverse_filtered_min_self_distances)).__round__(2),
                f'{overlap}reversefiltered_rmsds_below_2': (
                            100 * (reverse_filtered_rmsds < 2).sum() / len(reverse_filtered_rmsds)).__round__(2),
                f'{overlap}reversefiltered_rmsds_below_5': (
                            100 * (reverse_filtered_rmsds < 5).sum() / len(reverse_filtered_rmsds)).__round__(2),
                f'{overlap}reversefiltered_rmsds_percentile_25': np.percentile(reverse_filtered_rmsds, 25).round(2),
                f'{overlap}reversefiltered_rmsds_percentile_50': np.percentile(reverse_filtered_rmsds, 50).round(2),
                f'{overlap}reversefiltered_rmsds_percentile_75': np.percentile(reverse_filtered_rmsds, 75).round(2),

                f'{overlap}reversefiltered_centroid_below_2': (100 * (reverse_filtered_centroid_distances < 2).sum() / len(
                    reverse_filtered_centroid_distances)).__round__(2),
                f'{overlap}reversefiltered_centroid_below_5': (100 * (reverse_filtered_centroid_distances < 5).sum() / len(
                    reverse_filtered_centroid_distances)).__round__(2),
                f'{overlap}reversefiltered_centroid_percentile_25': np.percentile(reverse_filtered_centroid_distances,
                                                                                  25).round(2),
                f'{overlap}reversefiltered_centroid_percentile_50': np.percentile(reverse_filtered_centroid_distances,
                                                                                  50).round(2),
                f'{overlap}reversefiltered_centroid_percentile_75': np.percentile(reverse_filtered_centroid_distances,
                                                                                  75).round(2),
            })

            if N >= 5:
                top5_reverse_filtered_rmsds = np.min(
                    rmsds[np.arange(rmsds.shape[0])[:, None], reverse_confidence_ordering][:, :5], axis=1)
                top5_reverse_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                top5_reverse_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top5_reversefiltered_rmsds_below_2': (
                                100 * (top5_reverse_filtered_rmsds < 2).sum() / len(top5_reverse_filtered_rmsds)).__round__(
                        2),
                    f'{overlap}top5_reversefiltered_rmsds_below_5': (
                                100 * (top5_reverse_filtered_rmsds < 5).sum() / len(top5_reverse_filtered_rmsds)).__round__(
                        2),
                    f'{overlap}top5_reversefiltered_rmsds_percentile_25': np.percentile(top5_reverse_filtered_rmsds,
                                                                                        25).round(2),
                    f'{overlap}top5_reversefiltered_rmsds_percentile_50': np.percentile(top5_reverse_filtered_rmsds,
                                                                                        50).round(2),
                    f'{overlap}top5_reversefiltered_rmsds_percentile_75': np.percentile(top5_reverse_filtered_rmsds,
                                                                                        75).round(2),

                    f'{overlap}top5_reversefiltered_centroid_below_2': (
                                100 * (top5_reverse_filtered_centroid_distances < 2).sum() / len(
                            top5_reverse_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top5_reversefiltered_centroid_below_5': (
                                100 * (top5_reverse_filtered_centroid_distances < 5).sum() / len(
                            top5_reverse_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top5_reversefiltered_centroid_percentile_25': np.percentile(
                        top5_reverse_filtered_centroid_distances, 25).round(2),
                    f'{overlap}top5_reversefiltered_centroid_percentile_50': np.percentile(
                        top5_reverse_filtered_centroid_distances, 50).round(2),
                    f'{overlap}top5_reversefiltered_centroid_percentile_75': np.percentile(
                        top5_reverse_filtered_centroid_distances, 75).round(2),
                })

            if N >= 10:
                top10_reverse_filtered_rmsds = np.min(
                    rmsds[np.arange(rmsds.shape[0])[:, None], reverse_confidence_ordering][:, :10], axis=1)
                top10_reverse_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                top10_reverse_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                performance_metrics.update({
                    f'{overlap}top10_reversefiltered_rmsds_below_2': (100 * (top10_reverse_filtered_rmsds < 2).sum() / len(
                        top10_reverse_filtered_rmsds)).__round__(2),
                    f'{overlap}top10_reversefiltered_rmsds_below_5': (100 * (top10_reverse_filtered_rmsds < 5).sum() / len(
                        top10_reverse_filtered_rmsds)).__round__(2),
                    f'{overlap}top10_reversefiltered_rmsds_percentile_25': np.percentile(top10_reverse_filtered_rmsds,
                                                                                         25).round(2),
                    f'{overlap}top10_reversefiltered_rmsds_percentile_50': np.percentile(top10_reverse_filtered_rmsds,
                                                                                         50).round(2),
                    f'{overlap}top10_reversefiltered_rmsds_percentile_75': np.percentile(top10_reverse_filtered_rmsds,
                                                                                         75).round(2),

                    f'{overlap}top10_reversefiltered_centroid_below_2': (
                                100 * (top10_reverse_filtered_centroid_distances < 2).sum() / len(
                            top10_reverse_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top10_reversefiltered_centroid_below_5': (
                                100 * (top10_reverse_filtered_centroid_distances < 5).sum() / len(
                            top10_reverse_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top10_reversefiltered_centroid_percentile_25': np.percentile(
                        top10_reverse_filtered_centroid_distances, 25).round(2),
                    f'{overlap}top10_reversefiltered_centroid_percentile_50': np.percentile(
                        top10_reverse_filtered_centroid_distances, 50).round(2),
                    f'{overlap}top10_reversefiltered_centroid_percentile_75': np.percentile(
                        top10_reverse_filtered_centroid_distances, 75).round(2),
                })

    for k in performance_metrics:
        print(k, performance_metrics[k])

    if args.wandb:
        wandb.log(performance_metrics)
        histogram_metrics_list = [('rmsd', rmsds[:, 0]),
                                  ('centroid_distance', centroid_distances[:, 0]),
                                  ('mean_rmsd', rmsds.mean(axis=1)),
                                  ('mean_centroid_distance', centroid_distances.mean(axis=1))]
        if N >= 5:
            histogram_metrics_list.append(('top5_rmsds', top5_rmsds))
            histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))
        if N >= 10:
            histogram_metrics_list.append(('top10_rmsds', top10_rmsds))
            histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))
        if filtering_model is not None:
            histogram_metrics_list.append(('reverse_filtered_rmsds', reverse_filtered_rmsds))
            histogram_metrics_list.append(('reverse_filtered_centroid_distances', reverse_filtered_centroid_distances))
            histogram_metrics_list.append(('filtered_rmsd', filtered_rmsds))
            histogram_metrics_list.append(('filtered_centroid_distance', filtered_centroid_distances))
            if N >= 5:
                histogram_metrics_list.append(('top5_filtered_rmsds', top5_filtered_rmsds))
                histogram_metrics_list.append(('top5_filtered_centroid_distances', top5_filtered_centroid_distances))
                histogram_metrics_list.append(('top5_reverse_filtered_rmsds', top5_reverse_filtered_rmsds))
                histogram_metrics_list.append(
                    ('top5_reverse_filtered_centroid_distances', top5_reverse_filtered_centroid_distances))
            if N >= 10:
                histogram_metrics_list.append(('top10_filtered_rmsds', top10_filtered_rmsds))
                histogram_metrics_list.append(('top10_filtered_centroid_distances', top10_filtered_centroid_distances))
                histogram_metrics_list.append(('top10_reverse_filtered_rmsds', top10_reverse_filtered_rmsds))
                histogram_metrics_list.append(
                    ('top10_reverse_filtered_centroid_distances', top10_reverse_filtered_centroid_distances))
        os.makedirs(f'.plotly_cache/{cache_name}', exist_ok=True)
        images = []
        for metric_name, metric in histogram_metrics_list:
            d = {'EntropicBind': metric}
            df = pd.DataFrame(data=d)
            fig = px.ecdf(df, width=900, height=600, range_x=[0, 40])
            fig.add_vline(x=2, annotation_text='2 A;', annotation_font_size=20, annotation_position="top right",
                          line_dash='dash', line_color='firebrick', annotation_font_color='firebrick')
            fig.add_vline(x=5, annotation_text='5 A;', annotation_font_size=20, annotation_position="top right",
                          line_dash='dash', line_color='green', annotation_font_color='green')
            fig.update_xaxes(title=f'{metric_name} in Angstrom', title_font={"size": 20}, tickfont={"size": 20})
            fig.update_yaxes(title=f'Fraction of predictions with lower error', title_font={"size": 20},
                             tickfont={"size": 20})
            fig.update_layout(autosize=False, margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, plot_bgcolor='white',
                              paper_bgcolor='white', legend_title_text='Method', legend_title_font_size=17,
                              legend=dict(yanchor="bottom", y=0.1, xanchor="right", x=0.99, font=dict(size=17), ), )
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey')

            fig.write_image(os.path.join(f'.plotly_cache/{cache_name}', f'{metric_name}.png'))
            wandb.log({metric_name: wandb.Image(os.path.join(f'.plotly_cache/{cache_name}', f'{metric_name}.png'),
                                                caption=f"{metric_name}")})
            images.append(
                wandb.Image(os.path.join(f'.plotly_cache/{cache_name}', f'{metric_name}.png'), caption=f"{metric_name}"))
        wandb.log({'images': images})
