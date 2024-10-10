
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

__all__ = ["temporal_correlation", "temporal_correlation_difference"]

from typing import Iterable

import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


def temporal_correlation(
    graph_series: Iterable[Data],
) -> float:
    """
    Calculate temporal correlation within a graph sequence.

    Args:
        graph_series (Iterable[Data]): The graph sequence

    Returns:
        float: The temporal correlation coefficient.
    """

    dense_adjs = [
        to_dense_adj(g.edge_index, edge_attr=g.edge_attr, max_num_nodes=g.num_nodes)
        .cpu()
        .numpy()
        for g in graph_series
    ]

    c = np.zeros(len(dense_adjs) - 1)

    for t in range(len(dense_adjs) - 1):
        numerator = np.sum(dense_adjs[t] * dense_adjs[t + 1], axis=1).ravel()

        degrees_t = np.sum(dense_adjs[t], axis=1)
        degrees_t1 = np.sum(dense_adjs[t + 1], axis=1)

        denominator = np.sqrt(degrees_t * degrees_t1).ravel()

        nonzero_indices = np.nonzero(denominator)[0]

        n_t = np.sum(degrees_t != 0)
        n_t1 = np.sum(degrees_t1 != 0)

        c_m = np.sum(numerator[nonzero_indices] / denominator[nonzero_indices]) / max(
            n_t, n_t1
        )

        c[t] = c_m

    return float(np.mean(c))


def temporal_correlation_difference(
    original: Iterable[Data],
    generated: Iterable[Data],
) -> float:
    """
    Calculate temporal correlation difference between two graph sequences.

    Args:
        original (Iterable[Data]): The original graph sequence.
        generated (Iterable[Data]): The generated graph sequence.

    Returns:
        float: The temporal correlation difference.
    """
    return abs(temporal_correlation(original) - temporal_correlation(generated))
