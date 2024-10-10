import os
from copy import deepcopy
from typing import Optional, Union, Dict

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError,MetricCollection,KLDivergence
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import wandb


# from dgd.ggg_utils_deps import approx_small_symeig, our_small_symeig,extract_canonical_k_eigenfeat
# from dgd.ggg_utils_deps import  ensure_tensor, get_laplacian, asserts_enabled


class NoSyncMetricCollection(MetricCollection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching>>>>>>> main:utils.py

# Folders
def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('chains', exist_ok=True)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name, exist_ok=True)
        os.makedirs('chains/' + args.general.name, exist_ok=True)
    except OSError:
        pass


def to_dense(data, dataset_info, device=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    embed, _ = to_dense_batch(x=data.embed, batch=data.batch)
    embed = embed.float()
    assert embed.mean(dim=1).abs().max() < 1e-3
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)

    X, E = dataset_info.to_one_hot(X, E=E, node_mask=node_mask)

    y = X.new_zeros((X.shape[0], 0))

    if device is not None:
        X = X.to(device)
        E = E.to(device)
        y = y.to(device)
        embed = embed.to(device)
        node_mask = node_mask.to(device)

    data = PlaceHolder(X=X, embed=embed, E=E, y=y,  node_mask=node_mask)
    return data.mask()


class PlaceHolder:
    def __init__(self, embed, X, E, y, t_int=None, t=None, node_mask=None):
        self.embed = embed
        self.X = X
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.embed = self.embed.to(x.device) if self.embed is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        diag_mask = ~torch.eye(n, dtype=torch.bool,
                               device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.embed is not None:
            self.embed = self.embed * x_mask
            self.embed = self.embed - self.embed.mean(dim=1, keepdim=True)
        assert torch.allclose(self.E, torch.transembede(self.E, 1, 2))
        return self

    def collapse(self):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = - 1
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        return copy

    def __repr__(self):
        return (f"embed: {self.embed.shape if type(self.embed) == torch.Tensor else self.embed} -- " +
                f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- " +
                f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- " +
                f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}")


    def copy(self):
        return PlaceHolder(X=self.X, E=self.E, y=self.y, embed=self.embed, t_int=self.t_int, t=self.t,
                           node_mask=self.node_mask)


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'MolDiffusion_{cfg.dataset["name"]}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

