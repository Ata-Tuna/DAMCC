
### **
###  * Copyright (C) Euranova - All Rights Reserved 2024
###  * 
###  * This source code is protected under international copyright law.  All rights
###  * reserved and protected by the copyright holders.
###  * This file is confidential and only available to authorized individuals with the
###  * permission of the copyright holders.  If you encounter this file and do not have
###  * permission, please contact the copyright holders and delete this file at 
###  * research@euranova.eu
###  * It may only be used for academic research purposes as authorized by a written
###  * agreement issued by Euranova. 
###  * Any commercial use, redistribution, or modification of this software without 
###  * explicit written permission from Euranova is strictly prohibited. 
###  * By using this software, you agree to abide by the terms of this license. 
###  **

__all__ = ["get_nndr"]

from loguru import logger
import typing as ty
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningNodeData



def get_nndr(
    *args: Tensor, method: str = "l2", idx: int = 1
) -> Tuple[Tensor, Tensor, Tensor]:
    """Nearest Neighbor Distance Ratio.
    Args:
        *args (Tensor): Either a distance matrix (N,M), or two Tensors whoses sizes are (N,D) and (M,D).
        method (str):
            Distance type: L2
        idx (int):
            Index of the distance at the denominator.
    Returns:
        (Tuple[Tensor, Tensor, Tensor]):
            nndr: The NNDR (N,).
            hist: The histograms.
            edges: The bin edges.
    """
    assert isinstance(args[0], Tensor)
    # Distances
    if len(args) == 1:
        distances = args[0]
        # NNDR
        NNDR = _nndr(distances)
    else:
        assert isinstance(args[1], Tensor)
        x = args[0]
        y = args[1]
        method = method.lower()
        if method in ["l2"]:
            distances = torch.cdist(x, y)
        else:
            raise ValueError(
                f"Unknown method {method}. Please select valid distance type."
            )
    # NNDR
    NNDR = _nndr(distances, idx=idx)
    # Device
    device = distances.device
    # Hist and bin edges
    n_nodes_original = int(distances.size(0))
    hist = torch.Tensor([]).to(device)
    edges = torch.Tensor([]).to(device)
    for i in range(n_nodes_original):
        d = distances[i].view(-1)
        hi, ed = torch.histogram(d.to("cpu"), 100)
        hi, ed = hi.to(device), ed.to(device)
        hist = torch.cat((hist, hi), dim=0)
        edges = torch.cat((edges, ed), dim=0)
    return NNDR, hist, edges


def _nndr(distances: torch.Tensor, idx: int = 1) -> torch.Tensor:
    """NNDR.
    Args:
        distances (torch.Tensor): (N,M)
            Distance matrix, returned by `torch.cdist()`.
        idx (int):
            Index of the distance at the denominator.
    Returns:
        torch.Tensor: (N,)
            NNDR.
    """
    k = int(distances.size(1))
    top, _ = distances.detach().topk(k=k, dim=1, largest=False, sorted=True)
    if float(top.max()) < 1e-12:
        logger.warning(f"Distances are all zeros... {float(top.max())}")
    logger.trace(f"Top-k: {top}")
    nndr: torch.Tensor = top[:, 0] / top[:, idx]
    logger.trace(f"nndr: {nndr.size()}")
    # For each row, if the nndr is NaN, it means that the two closest distances are 0
    nndr[nndr.isnan()] = 1
    return nndr
