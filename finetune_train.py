import copy
import math
import os
import shutil
import pickle
from functools import partial
from argparse import Namespace
from rdkit import Chem

import wandb
import torch
import numpy as np
import rdkit.Chem as Chem

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rdkit.Chem import AllChem, RemoveHs, RemoveAllHs, MolFromSmiles
from torch_geometric.data import HeteroData

import random
import traceback

from datasets.loader import CombineDatasets
from datasets.process_mols import generate_conformer, get_lig_graph
from utils.gnina_utils import get_gnina_poses, invert_permutation
from utils.molecules_utils import get_symmetry_rmsd

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from utils.training import train_epoch, test_epoch, loss_function
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage, remove_all_hs
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule, get_inverse_schedule
from utils.torsion import get_transformation_mask

from datasets.pdbbind import NoiseTransform
from bootstrapping.buffer import CBBuffer
from datasets.moad import MOAD
from datasets.dataloader import DataLoader, DataListLoader

from confidence.dataset import ListDataset
from bootstrapping.parsing import parse_cb_args


def get_filtering_dataset(args, model_args):
    dataset = MOAD(transform=None, root=args.moad_dir, limit_complexes=args.limit_complexes,
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
                    esm_embeddings_path=args.moad_esm_embeddings_path,
                    esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                    require_ligand=True,
                    num_workers=args.num_workers,
                    knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                    include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                    num_conformers=1,
                    unroll_clusters=True,
                    min_ligand_size=args.min_ligand_size,
                    max_receptor_size=args.max_receptor_size,
                    remove_promiscuous_targets=args.remove_promiscuous_targets)
    return dataset

def construct_datasets(args, t_to_sigma):
    transform_finetune = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.buffer_sampling_alpha, beta=args.buffer_sampling_beta,
                               rot_alpha=args.rot_alpha, rot_beta=args.rot_beta, tor_alpha=args.tor_alpha,
                               tor_beta=args.tor_beta, separate_noise_schedule=args.separate_noise_schedule,
                               asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                               include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                               minimum_t=args.minimum_t, sampling_mixing_coeff=args.sampling_mixing_coeff)

    common_args = {'limit_complexes': args.limit_complexes,
                           'receptor_radius': args.receptor_radius,
                           'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                           'remove_hs': args.remove_hs, 'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                           'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                           'knn_only_graph': not args.not_knn_only_graph}

    finetune_dataset = CBBuffer(transform=transform_finetune,
                                           cluster_name=args.cb_cluster,
                                           multiplicity=args.train_multiplicity,
                                           max_complexes_per_couple=args.max_complexes_per_couple,
                                           fixed_length=args.fixed_length,
                                           temperature=args.temperature,
                                           buffer_decay=args.buffer_decay,
                                           reset_buffer=args.reset_buffer)

    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               rot_alpha=args.rot_alpha, rot_beta=args.rot_beta, tor_alpha=args.tor_alpha,
                               tor_beta=args.tor_beta, separate_noise_schedule=args.separate_noise_schedule,
                               asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                               include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms)

    target_dataset = MOAD(cache_path=args.cache_path, split=args.split, single_cluster_name=args.cb_cluster, 
                          keep_original=True, multiplicity=args.target_multiplicity, max_receptor_size=args.max_receptor_size,
                          remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                          esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                          unroll_clusters=True, root=args.moad_dir, transform=transform,
                          esm_embeddings_path=args.moad_esm_embeddings_path, require_ligand=True, **common_args)

    if args.keep_original_train:
        train_dataset = MOAD(cache_path=args.cache_path, split='train', transform=transform,
                              keep_original=True, multiplicity=args.train_multiplicity,
                              max_receptor_size=args.max_receptor_size,
                              remove_promiscuous_targets=args.remove_promiscuous_targets,
                              min_ligand_size=args.min_ligand_size,
                              esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                              unroll_clusters=True, root=args.moad_dir,
                              esm_embeddings_path=args.moad_esm_embeddings_path, require_ligand=True,
                             total_dataset_size=args.total_trainset_size, **common_args)
        finetune_dataset = CombineDatasets(finetune_dataset, train_dataset)

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    target_loader = loader_class(dataset=target_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=False, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    return finetune_dataset, target_loader


def inference_epoch(model, filtering_model, complex_graphs, filtering_complex_dict, device, t_to_sigma, args, filtering_args, confidence_cutoff):
    # Run inference and confidence model, return inference metrics and generated complexes above confidence cutoff
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    if args.asyncronous_noise_schedule:
        tr_schedule = get_inverse_schedule(t_schedule, args.sampling_alpha, args.sampling_beta)
        rot_schedule = get_inverse_schedule(t_schedule, args.rot_alpha, args.rot_beta)
        tor_schedule = get_inverse_schedule(t_schedule, args.tor_alpha, args.tor_beta)
    else:
        tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds, top_rmsds, gnina_rmsds_list = [], [], [], []
    complexes_to_keep = []
    confidences_list = []
    
    for orig_complex_graph in tqdm(loader):
        torch.cuda.empty_cache()
        if filtering_model is not None and not (
                filtering_args.use_original_model_cache or filtering_args.transfer_weights) and orig_complex_graph.name[0] not in filtering_complex_dict.keys():
            print(
                f"HAPPENING | The filtering dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
            continue
        
        if filtering_model is not None and not (
                filtering_args.use_original_model_cache or filtering_args.transfer_weights):
            filtering_complex = filtering_complex_dict[orig_complex_graph.name[0]]
            filtering_data_list = [copy.deepcopy(filtering_complex) for _ in
                                    range(args.inference_samples)]

        else:
            filtering_data_list = None
        
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max,
                           pocket_knowledge=args.inf_pocket_knowledge, pocket_cutoff=args.inf_pocket_cutoff)
        
        predictions_list = None
        confidences = None
        failed_convergence_counter = 0
        bs = args.inference_batch_size
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                     inference_steps=args.inference_steps,
                                                     tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                     tor_schedule=tor_schedule,
                                                     device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                     confidence_model=filtering_model,
                                                     filtering_data_list=filtering_data_list,
                                                     filtering_model_args=filtering_args,
                                                     asyncronous_noise_schedule=args.asyncronous_noise_schedule,
                                                     t_schedule=t_schedule, batch_size=bs)
            except Exception as e:
                failed_convergence_counter += 1
                if bs > 1:
                    bs = bs // 2
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
                traceback.print_exc()
        
        if failed_convergence_counter > 5: continue
        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())
            
        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()
        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()

        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

        if confidences is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
            confidences = confidences[:, 0]
        top_rmsds.append(rmsd[confidences.argmax()])
        confidences_list.extend([c.detach().cpu().item() for c in confidences])

        if args.oracle_confidence:
            confidences = - 4 * np.tanh(2 * rmsd / 3 - 2)

        if args.gnina_minimize:
            print('Running gnina on all predicted ligand positions for energy minimization.')
            gnina_rmsds, gnina_scores = [], []
            lig = copy.deepcopy(orig_complex_graph.mol[0])
            positions = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list])
            gnina_positions = []

            conf = confidences
            if conf is not None:
                conf = conf.cpu().numpy()
                conf = np.nan_to_num(conf, nan=-1e-6)
                re_order = np.argsort(conf)[::-1]
                positions = positions[re_order]
                predictions_list = [predictions_list[i] for i in re_order]
    
            # Run the subprocesses in parallel
            run_dir = os.path.join(args.log_dir, args.run_name)
            gnina_logs_dir = os.path.join(run_dir, "gnina_logs")
            args.out_dir = run_dir
            
            if not os.path.exists(gnina_logs_dir):
                print(f'Make gnina logs dir: {gnina_logs_dir}')
                os.makedirs(gnina_logs_dir)
                
            if args.gnina_parallel:
                print('Running gnina subprocesses in parallel')
                input_list = [(args, lig, pos, orig_complex_graph.original_center.cpu().numpy(), orig_complex_graph.name[0], tid) for tid, pos in enumerate(positions[:args.gnina_poses_to_optimize])]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    gnina_output_list = list(executor.map(lambda params: get_gnina_poses(*params), input_list))
            else:
                print('Running gnina subprocesses sequentially')
                gnina_output_list = [get_gnina_poses(args, lig, pos,
                                       orig_complex_graph.original_center.cpu().numpy(),
                                       orig_complex_graph.name[0]) for pos in positions[:args.gnina_poses_to_optimize]]
            
            # for pos in positions[:args.gnina_poses_to_optimize]:
            #     gnina_ligand_pos, gnina_mol, gnina_score = get_gnina_poses(args, lig, pos,
            #                                                                orig_complex_graph.original_center.cpu().numpy(),
            #                                                                orig_complex_graph.name[0])
            
            # Process gnina RMSDs
            for idx in range(args.gnina_poses_to_optimize):
                pos = positions[idx]
                gnina_ligand_pos, gnina_mol, gnina_score = gnina_output_list[idx]
                _rmsds = []
                for i in range(len(orig_ligand_pos)):
                    try:
                        rmsd, (idx1, idx2) = get_symmetry_rmsd(mol, orig_ligand_pos[i], gnina_ligand_pos, gnina_mol, return_permutation=True)
                        inv_idx1 = invert_permutation(idx1)
                        gnina_ligand_pos = gnina_ligand_pos[idx2][inv_idx1]
                        test_rmsd = np.sqrt(((gnina_ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=1).mean(axis=0))
                        # print(rmsd, test_rmsd, "should be close")
                    except Exception as e:
                        print("Using non corrected RMSD because of the error:", e, "and giving back the score model position")
                        print(Chem.MolToSmiles(mol))
                        print(Chem.MolToSmiles(gnina_mol))
                        rmsd = np.sqrt(((gnina_ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=1).mean(axis=0))
                        gnina_ligand_pos = pos
                    _rmsds.append(rmsd)
                _rmsds = np.asarray(_rmsds)
                rmsd = np.min(_rmsds, axis=0)
                gnina_rmsds.append(rmsd)
                gnina_scores.append(gnina_score)
                gnina_positions.append(gnina_ligand_pos)

            gnina_rmsds = np.asarray(gnina_rmsds)
            gnina_scores = np.asarray(gnina_scores)
            gnina_rmsds_list.extend([r for r in gnina_rmsds])

            for i in range(len(gnina_positions)):
                if gnina_scores[i] > confidence_cutoff:
                    predictions_list[i]['ligand'].pos = torch.from_numpy(gnina_positions[i]).float().to(device)
                    complexes_to_keep.append((predictions_list[i], gnina_scores[i]))

        else:
            complexes_to_keep.extend([(predictions_list[i], confidences[i]) for i in range(args.inference_samples) if confidences[i] > confidence_cutoff])
    
    rmsds = np.array(rmsds)
    gnina_rmsds = np.array(gnina_rmsds_list) if args.gnina_minimize else None
    min_rmsds = np.array(min_rmsds)
    top_rmsds = np.array(top_rmsds)
    confidences_list = np.array(confidences_list)

    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'gnina_rmsds_lt2': (100 * (gnina_rmsds < 2).sum() / len(gnina_rmsds)) if args.gnina_minimize else None,
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'filtered_rmsds_lt2': (100 * (top_rmsds < 2).sum() / len(min_rmsds)),
              'filtered_rmsds_lt5': (100 * (top_rmsds < 5).sum() / len(min_rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),
              'avg_confidence': confidences_list.mean(),
              'median_confidence': np.median(confidences_list)}
    
    print(f'Complexes to keep from inference: {len(complexes_to_keep)}')
    return losses, complexes_to_keep, top_rmsds


def inference_finetune(args, model, filtering_model, filtering_args, filtering_complex_dict, confidence_cutoff, 
                       optimizer, ema_weights, finetune_dataset, target_loader, t_to_sigma, run_dir):
    
    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                          tor_weight=args.tor_weight, no_torsion=args.no_torsion, backbone_weight=args.backbone_loss_weight,
                          sidechain_weight=args.sidechain_loss_weight)
    
    finetune_loader = None
    if args.save_metrics:
        metrics = {}
    
    filtered_rmsds = None
    
    print("Starting inference-finetuning...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: print("Run name: ", args.run_name)
        logs = {}
         
        ema_weights.store(model.parameters())

        # load ema parameters into model for running inference
        if args.use_ema: ema_weights.copy_to(model.parameters()) 

        if epoch % args.cb_inference_freq == 0:
            print("Doing inference and saving complexes to finetuning dataset.")
            inf_dataset = [target_loader.dataset.get(i) for i in range(min(args.num_inference_complexes, target_loader.dataset.__len__()))]

            complexes = []
            inf_metrics = None
            iterations = args.initial_iterations if epoch == 0 else args.inference_iterations
            
            for i in range(iterations):
                inf_m, compl, filtered_rmsds = inference_epoch(model, filtering_model, inf_dataset, filtering_complex_dict, device, t_to_sigma, args,
                                                            filtering_args, confidence_cutoff)
                if inf_metrics is None: inf_metrics = {k:[] for k in inf_m if inf_m[k] is not None}
                for k in inf_metrics: inf_metrics[k].append(inf_m[k])
                complexes.extend(compl)

            for k in inf_metrics: 
                try:
                    inf_metrics[k] = np.mean(inf_metrics[k])
                except Exception as e:
                    inf_metrics[k] = None
            
                
            # update finetune_dataset and construct new finetune_loader
            finetune_dataset.add_complexes(complexes)
            loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
            finetune_loader = loader_class(dataset=finetune_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)

            print("Epoch {}: Target inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} min_rmsds_lt2 {:.3f} min_rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], inf_metrics['min_rmsds_lt2'], inf_metrics['min_rmsds_lt5']))
            logs.update({'targetinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if not args.use_ema: ema_weights.copy_to(model.parameters())
        ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
        ema_weights.restore(model.parameters())
        
        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

        if not (args.save_model_freq is None) and (epoch + 1) % args.save_model_freq == 0:
            
            torch.save(state_dict, os.path.join(run_dir, f'epoch{epoch+1}_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, f'epoch{epoch+1}_ema_inference_epoch_model.pt'))


        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

        train_losses = train_epoch(model, finetune_loader, optimizer, device, t_to_sigma, loss_fn, 
                                   ema_weights, torsional=False)
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   sc {:.4f}  lr {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['sidechain_loss'], optimizer.param_groups[0]['lr']))

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        if args.save_metrics:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            for k, v in logs.items():
                if k in metrics:
                    metrics[k].append(v)
                else:
                    metrics[k] = [v]

    if args.save_metrics:
        with open(os.path.join(run_dir, 'training_metrics.pkl'), 'wb') as file:
            pickle.dump(metrics, file)
        if args.save_final_rmsds:
            np.save(os.path.join(run_dir, 'final_filtered_rmsds.npy'), filtered_rmsds)
            
def main_function():
    args = parse_cb_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.wandb:
        wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
   
    # Construct datasets
    t_to_sigma = partial(t_to_sigma_compl, args=args)

    finetune_dataset, target_loader = construct_datasets(args, t_to_sigma)
    
    # Load pretrained score model
    assert args.pretrain_dir is not None
    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, _ = get_optimizer_and_scheduler(args, model, scheduler_mode='min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)

    dict = torch.load(f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt', map_location=torch.device('cpu'))
    model.module.load_state_dict(dict, strict=True)
    print("Using pretrained model", f'{args.pretrain_dir}/{args.pretrain_ckpt}.pt')

    numel = sum([p.numel() for p in model.parameters()])
    print('SUCCESS| Score Model with', numel, 'parameters')

    # Loading confidence (filtering) model
    assert args.filtering_model_dir is not None

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

    filtering_complex_dict = None

    if not (filtering_args.use_original_model_cache or filtering_args.transfer_weights):
        # if the filtering model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
        print('HAPPENING | filtering model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the filtering model now.')
        filtering_test_dataset = get_filtering_dataset(args, filtering_args)
        filtering_complex_dict = filtering_test_dataset.get_all_complexes()

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
                                confidence_mode=True)
    state_dict = torch.load(f'{args.filtering_model_dir}/{args.filtering_ckpt}', map_location=torch.device('cpu'))
    filtering_model.load_state_dict(state_dict, strict=True)

    numel = sum([p.numel() for p in filtering_model.parameters()])
    print('SUCCESS| Confidence Model with', numel, 'parameters')

    filtering_model = filtering_model.to(device)
    filtering_model.eval()

    
    if args.wandb:
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    inference_finetune(args, model, filtering_model, filtering_model_args, filtering_complex_dict, args.confidence_cutoff,
                       optimizer, ema_weights, finetune_dataset, target_loader, t_to_sigma, run_dir)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
