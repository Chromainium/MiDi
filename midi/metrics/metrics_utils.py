import math
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from midi.datasets.dataset_utils import Statistics
from torchmetrics import MeanAbsoluteError


def molecules_to_datalist(trees):
    data_list = []
    for tree in trees:
        x = tree.mol_types.long()
        rxns = tree.rxn_types.long()
        embeddings = tree.embeddings
        edge_index = rxns.nonzero().contiguous().T
        rxn_types = rxn[edge_index[0], edge_index[1]]
        edge_attr = rxn_types.long()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, embed=embeddings)
        data_list.append(data)

    return data_list


def compute_all_statistics(data_list, atom_encoder):
    num_nodes = node_counts(data_list)
    mol_types = mol_type_counts(data_list, num_classes=len(atom_encoder))
    print(f"Atom types: {atom_types}")
    bond_types = edge_counts(data_list)
    print(f"Bond types: {bond_types}")
    return Statistics(num_nodes=num_nodes, atom_types=atom_types, bond_types=bond_types)


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def mol_type_counts(data_list, num_classes):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()

    counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = torch.nn.functional.one_hot(data.edge_attr - 1, num_classes=num_bond_types - 1).sum(dim=0).numpy()
        d[0] += num_non_edges
        d[1:] += edge_types

    d = d / d.sum()
    return d


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) == int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
        """ preds and target are 1d tensors. They contain histograms for bins that are regularly spaced """
        target = normalize(target) / step_size
        preds = normalize(preds) / step_size
        max_len = max(len(preds), len(target))
        preds = F.pad(preds, (0, max_len - len(preds)))
        target = F.pad(target, (0, max_len - len(target)))

        cs_target = torch.cumsum(target, dim=0)
        cs_preds = torch.cumsum(preds, dim=0)
        return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert target.dim() == 1 and preds.shape == target.shape, f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s
