import pickle

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def tree_to_torch_geometric(tree, mol_encoder):
    mol_types, adj = tree.get_adjacency()
    adj = torch.from_numpy(adj)
    edge_index = adj.nonzero().contiguous().T
    rxn_types = adj[edge_index[0], edge_index[1]]
    edge_attr = rxn_types.long()

    embed = torch.tensor([mol.data['embedding'] for mol in mol_types])
    embed = embed - torch.mean(embed, dim=0, keepdim=True)

    mol_types = torch.Tensor(mol_types).long()

    data = Data(x=mol_types, edge_index=edge_index, edge_attr=edge_attr, embed=embed)
    return data


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, mol_types, rxn_types):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.mol_types = mol_types
        self.rxn_types = rxn_types
