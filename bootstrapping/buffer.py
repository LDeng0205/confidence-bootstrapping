import torch
from torch_geometric.data import Dataset

import os
import pickle
import numpy as np
import copy

class CBBuffer(Dataset):
    def __init__(self, cluster_name=None, root=None, transform=None, multiplicity=1, max_complexes_per_couple=None, fixed_length=None, temperature=1.0, buffer_decay=0.2, reset_buffer=False):
        super(CBBuffer, self).__init__(root, transform)

        self.multiplicity = multiplicity
        self.complexes = []
        self.iteration = 0
        self.max_complexes_per_couple = max_complexes_per_couple
        self.fixed_length = fixed_length
        self.temperature = temperature
        self.buffer_decay = buffer_decay
        self.reset_buffer = reset_buffer

        with open("data/BindingMOAD_2020_processed/new_cluster_to_ligands.pkl", "rb") as f:
            self.cluster_to_ligands = pickle.load(f)

        assert cluster_name is not None

        self.ligand_names = self.cluster_to_ligands[cluster_name]
        
        print(f'There are {len(self.ligand_names)} complexes in the cluster {cluster_name}')

        self.ligand_cnt = {ligand_name: 0 for ligand_name in self.ligand_names} # Dictionary to keep track of the number of each complex
        
        self.print_statistics()
        print('SUCCESS| Buffer initialized.')

    def get(self, idx):
        if self.fixed_length is None:
            complex_graph = copy.deepcopy(self.complexes[idx % len(self.complexes)])
        else:
            confidences = np.asarray([complex_graph.confidence for complex_graph in self.complexes])
            weights = np.exp(confidences * self.temperature)
            weights = weights / np.sum(weights)
            idx = np.random.choice(len(self.complexes), p=weights)
            complex_graph = copy.deepcopy(self.complexes[idx])

        for a in ['confidence', 'iteration']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)
            if hasattr(complex_graph['ligand'], a):
                delattr(complex_graph['ligand'], a)

        return complex_graph
    
    def len(self):
        return len(self.complexes) * self.multiplicity if self.fixed_length is None else self.fixed_length
    
    def print_statistics(self):
        # Prints how many of each complexes is contained in the dataset
        print(f'Buffer with {len(self.complexes)} complexes.')
        for ligand, cnt in self.ligand_cnt.items():
            print(f'Ligand: {ligand} Cnt: {cnt}')

    def add_complexes(self, new_complex_list):
        print(f'Adding {len(new_complex_list)} new complexes to the Buffer dataset.')
        for complex_graph, confidence in new_complex_list:
            complex_graph.confidence = confidence
            complex_graph.iteration = self.iteration
            t = 0
            t_value = {'tr': t * torch.ones(1), 'rot': t * torch.ones(1), 'tor': t * torch.ones(1)}
        
            lig_node_t =  {'tr': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'rot': t * torch.ones(complex_graph['ligand'].num_nodes),
                                            'tor': t * torch.ones(complex_graph['ligand'].num_nodes)}
            rec_node_t = {'tr': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'rot': t * torch.ones(complex_graph['receptor'].num_nodes),
                                                'tor': t * torch.ones(complex_graph['receptor'].num_nodes)}
            
            complex_graph.complex_t = t_value
            complex_graph['ligand'].node_t = lig_node_t
            complex_graph['receptor'].node_t = rec_node_t

            # update ligand_cnt dictionary
            self.ligand_cnt[complex_graph.name[0]] += 1
            complex_graph = complex_graph.cpu()

        self.iteration += 1
        
        if self.reset_buffer:
            self.complexes = [c for c, _ in new_complex_list]
        else:
            self.complexes.extend([c for c, _ in new_complex_list])

        print(f'There are now {len(self.complexes)} complexes in the buffer.')
        if self.max_complexes_per_couple is not None:
            c_to_samples = {}
            for s in self.complexes:
                c_to_samples[s.name[0][:6]] = []

            for s in self.complexes:
                c_to_samples[s.name[0][:6]].append((s.confidence + self.buffer_decay * s.iteration, s)) # the policy is quite arbitrary here

            # Sort complexes by confidence and iteration and keep only the top ones
            for c in c_to_samples:
                if len(c_to_samples[c]) > self.max_complexes_per_couple:
                    c_to_samples[c] = sorted(c_to_samples[c], key=lambda x: x[0], reverse=True)
                    c_to_samples[c] = c_to_samples[c][:self.max_complexes_per_couple]

            self.complexes = []
            for c in c_to_samples:
                for _, s in c_to_samples[c]:
                    self.complexes.append(s)
            print(f'After filtering {len(self.complexes)} complexes in the buffer.')

        self.print_statistics()
