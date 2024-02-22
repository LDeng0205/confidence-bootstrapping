import glob
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
import os.path as osp
from datasets.process_mols import get_lig_graph_with_matching
from utils.diffusion_utils import modify_conformer, get_inverse_schedule
from utils.torsion import modify_conformer_torsion_angles
from utils.utils import read_strings_from_txt
from utils import so3, torus


dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')


class TorsionalNoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, alpha=1, beta=1, tor_alpha=1, tor_beta=1, separate_noise_schedule=False, asyncronous_noise_schedule=False):
        self.t_to_sigma = t_to_sigma
        self.alpha, self.beta = alpha, beta
        self.tor_alpha = tor_alpha
        self.tor_beta = tor_beta
        self.separate_noise_schedule = separate_noise_schedule
        self.asyncronous_noise_schedule = asyncronous_noise_schedule

    def __call__(self, data):

        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        if self.separate_noise_schedule:
            t_tr = np.random.uniform(0, 1)
            t_rot = np.random.uniform(0, 1)
            t_tor = np.random.beta(self.tor_alpha, self.tor_beta)
        elif self.asyncronous_noise_schedule:
            t = np.random.uniform(0, 1)
            t_tr = t
            t_rot = t
            t_tor = get_inverse_schedule(t, self.tor_alpha, self.tor_beta)
        else:
            t = np.random.beta(self.alpha, self.beta)
            t_tr, t_rot, t_tor = t,t,t

        tor_sigma = self.t_to_sigma(t_tor)
        data['ligand'].node_t = {'tr': t_tr * torch.ones(data['ligand'].num_nodes), 'rot': t_rot * torch.ones(data['ligand'].num_nodes),'tor': t_tor * torch.ones(data['ligand'].num_nodes)}
        data.complex_t = {'tr': t_tr * torch.ones(1), 'rot': t_rot * torch.ones(1),'tor': t_tor * torch.ones(1)}

        if self.asyncronous_noise_schedule:
            data['ligand'].node_t['t'] = t * torch.ones(data['ligand'].num_nodes)
            data.complex_t['t'] = t * torch.ones(1)

        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum())
        data['ligand'].pos = modify_conformer_torsion_angles(data['ligand'].pos, data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                             data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else
                                                             data['ligand'].mask_rotate[0], torsion_updates).to(data['ligand'].pos.device)

        data.tor_score = torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        return data


class Torsional(Dataset):
    def __init__(self, root, mode, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0, num_workers=1,
                 multiplicity=1, popsize=15, maxiter=15, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1,
                 matching_tries=1):

        super(Torsional, self).__init__(root, transform)
        self.root = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.mode = mode
        cache_path += '_geom'
        self.full_cache_path = os.path.join(cache_path, f'{mode}_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}' +
                                                        '' if matching_tries == 1 else f'matchingTries{matching_tries}')
        self.popsize, self.maxiter = popsize, maxiter
        self.keep_original = keep_original
        self.num_conformers = num_conformers
        self.matching_tries = matching_tries

        if not os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()

        print('loading data from memory')
        with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
            self.complex_graphs = pickle.load(f)
        with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
            self.rdkit_ligands = pickle.load(f)

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
        return complex_graph

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        split_idx = 0 if self.mode == 'train' else 1 if self.mode == 'val' else 2
        split = sorted(np.load(self.split_path, allow_pickle=True)[split_idx])
        if self.limit_complexes:
            split = split[:self.limit_complexes]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]
        smiles = [smi[len(self.root):-7] for smi in smiles]

        print(f'Loading {len(smiles)} complexes.')

        if self.num_workers > 1:
            for i in range(len(smiles)//1000+1):
                complex_names = smiles[1000*i:1000*(i+1)]
                complex_graphs, rdkit_ligands = [], []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                with tqdm(total=len(complex_names), desc=f'loading ligands {i}/{len(smiles)//1000+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_complex, complex_names):
                        if t:
                            complex_graphs.append(t[0])
                            rdkit_ligands.append(t[1])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)

                with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                    pickle.dump((complex_graphs), f)
                with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                    pickle.dump((rdkit_ligands), f)

            complex_graphs_all = []
            for i in range(len(smiles)//1000+1):
                with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'rb') as f:
                    l = pickle.load(f)
                    complex_graphs_all.extend(l)
            with open(os.path.join(self.full_cache_path, f"heterographs.pkl"), 'wb') as f:
                pickle.dump((complex_graphs_all), f)

            rdkit_ligands_all = []
            for i in range(len(smiles) // 1000 + 1):
                with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                    l = pickle.load(f)
                    rdkit_ligands_all.extend(l)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands_all), f)
        else:
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(smiles), desc='loading ligands') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_complex, smiles):
                    if t:
                        complex_graphs.append(t[0])
                        rdkit_ligands.append(t[1])
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

    def get_complex(self, smile):
        print(smile)
        if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
            print(os.path.join(self.root, smile + '.pickle'))
            print("Error", smile, 'raw_pickle_not_found')
            return False

        pickle_file = osp.join(self.root, smile + '.pickle')
        with open(pickle_file, "rb") as f:
            mol_dic = pickle.load(f)

        smile = mol_dic['smiles']

        if '.' in smile:
            print("Error", smile, 'dot_in_smile')
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            print("Error", smile, 'mol_from_smiles_failed')
            return False

        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            print("Error", smile, 'no_substruct_match')
            return False

        if N < 4:
            print("Error", smile, 'mol_too_small')
            return False

        if self.remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=False)

        if self.max_lig_size != None and mol.GetNumAtoms() > self.max_lig_size:
            print(f'Ligand of size {mol.GetNumAtoms()} is larger than max_lig_size {self.max_lig_size}. Not including {smile} in preprocessed data.')
            return False
        complex_graph = HeteroData()
        complex_graph['name'] = smile

        try:
            get_lig_graph_with_matching(mol_=mol, complex_graph=complex_graph, popsize=self.popsize, maxiter=self.maxiter,
                                        matching=True, keep_original=self.keep_original, num_conformers=self.num_conformers,
                                        remove_hs=self.remove_hs, tries=self.matching_tries)

        except Exception as e:
            print(e)
            return None

        return complex_graph, mol