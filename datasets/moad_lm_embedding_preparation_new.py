import os
import pickle
from argparse import FileType, ArgumentParser

import numpy as np
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/BindingMOAD_2020_ab_processed_biounit/pdb_protein/', help='')
parser.add_argument('--out_file', type=str, default="../../distillation/moad_mapping.pkl")
args = parser.parse_args()

data_dir = args.data_dir
names = os.listdir(data_dir)
names = [n[:6] for n in names]
#%%
from Bio import SeqIO
biopython_parser = PDBParser()

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

name_to_sequence = {}
for name in tqdm(names):
    if name == '.DS_Store': continue
    if not os.path.exists(os.path.join(data_dir, f'{name}_protein.pdb')):
        print(f"We are skipping {name} because there was no {name}_protein.pdb")
        continue
    rec_path = os.path.join(data_dir, f'{name}_protein.pdb')
    try:
        structure = biopython_parser.get_structure('random_id', rec_path)
    except Exception as e:
        print("encountered error while parsing ", name, " with error: ", e)
        continue
    structure = structure[0]
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', name, '. Replacing it with a dash - .')
        name_to_sequence[name + '_chain_' + str(i)] = seq

# save to file
with open(args.out_file, 'wb') as f:
    pickle.dump(name_to_sequence, f)