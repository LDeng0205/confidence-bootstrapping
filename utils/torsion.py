import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from utils.geometry import rigid_transform_Kabsch_independent_torch, axis_angle_to_matrix

"""
    Preprocessing and computation for torsional updates to conformers
"""


def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(min(edges.shape[0], len(G.edges()))):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    
    if type(mask_rotate) == list: mask_rotate = mask_rotate[0]
        
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        if mask_rotate[idx_edge, u] or (not mask_rotate[idx_edge, v]):
            print("mask rotate exception")
        #assert not mask_rotate[idx_edge, u]
        #assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def modify_conformer_torsion_angles_batch(pos, edge_index, mask_rotate, torsion_updates):
    pos = pos + 0
    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[:, u] - pos[:, v]  # convention: positive rotation if pointing inwards
        rot_mat = axis_angle_to_matrix(
            rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * torsion_updates[:, idx_edge:idx_edge + 1])

        pos[:, mask_rotate[idx_edge]] = torch.bmm(pos[:, mask_rotate[idx_edge]] - pos[:, v:v + 1], torch.transpose(rot_mat, 1, 2)) + pos[:, v:v + 1]

    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new


def get_dihedrals(data_list):
    edge_index, edge_mask = data_list[0]['ligand', 'ligand'].edge_index, data_list[0]['ligand'].edge_mask
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))
    # dihedral_numpy = np.asarray(dihedral)
    # print(dihedral_numpy.shape)
    dihedral = torch.tensor(dihedral)
    return dihedral


def bdot(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)


def get_torsion_angles(dihedral, batch_pos):
    c, a, b, d = dihedral[:, 0], dihedral[:, 1], dihedral[:, 2], dihedral[:, 3]
    c_project_ab = batch_pos[:,a] + bdot(batch_pos[:,c] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    d_project_ab = batch_pos[:,a] + bdot(batch_pos[:,d] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    dshifted = batch_pos[:,d] - d_project_ab + c_project_ab
    cos = bdot(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab) / (
                torch.norm(dshifted - c_project_ab, dim=-1, keepdim=True) * torch.norm(batch_pos[:,c] - c_project_ab, dim=-1,
                                                                                       keepdim=True))
    cos = torch.clamp(cos, -1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cos)
    #assert not torch.any(torch.isnan(angle)), (angle, cos)
    sign = torch.sign(bdot(torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab), batch_pos[:,b] - batch_pos[:,a]))
    torsion_angles = (angle * sign).squeeze(-1)

    #assert torch.all(torsion_angles > (- np.pi - 0.01)) and torch.all(torsion_angles < (np.pi + 0.01))
    return torsion_angles


def get_torsion_angles_svgd(dihedral, batch_pos):
    tau = get_torsion_angles(dihedral, batch_pos)
    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
    tau_diff = torch.fmod(tau_diff + 3 * np.pi, 2 * np.pi) - np.pi
    assert torch.all(tau_diff < np.pi + 0.1) and torch.all(tau_diff > -np.pi - 0.1), tau_diff
    tau_matrix = torch.sum(tau_diff ** 2, dim=-1, keepdim=True)

    return tau_matrix, tau_diff


def get_rigid_svgd(batch_pos):
    n = len(batch_pos)
    tr_diff, rot_diff = torch.zeros(n, n, 3), torch.zeros(n, n, 3)

    for i in range(n-1):
        for j in range(i+1, n):
            t, R_vec = rigid_transform_Kabsch_independent_torch(batch_pos[i].T, batch_pos[j].T)
            tr_diff[i][j], tr_diff[j][i] = t.squeeze(1), -t.squeeze(1)
            rot_diff[i][j], rot_diff[j][i] = R_vec, -R_vec

    tr_matrix = torch.sum(tr_diff ** 2, dim=-1, keepdim=True)
    rot_matrix = torch.sum(rot_diff ** 2, dim=-1, keepdim=True)
    return tr_matrix, rot_matrix, tr_diff, rot_diff


