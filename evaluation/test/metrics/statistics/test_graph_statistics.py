
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

import networkx as nx
import numpy as np
import torch
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
import pytest
from brainiac_temporal.metrics.statistics import GraphStatistics, normalize_hist


def test_graph_statistics() -> None:
    """Tests the GraphStatistics class."""
    graph = nx.Graph(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5), (1, 4)]
    )

    graph_statistics = GraphStatistics("all", to_tensors=False)([graph])

    for sequence in graph_statistics.values():
        for v in sequence:
            assert isinstance(v, (np.ndarray, float, int))


def test_graph_statistics_on_single_graph() -> None:
    """Tests the GraphStatistics class on a single graph."""
    graph = nx.Graph(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5), (1, 4)]
    )

    graph_statistics = GraphStatistics("all", to_tensors=True)(graph)

    for sequence in graph_statistics.values():
        assert len(sequence) == 1
        assert isinstance(sequence[0], (torch.Tensor))


def test_graph_statistics_on_tgt() -> None:
    """Tests the GraphStatistics class on a torch_geometric_temporal dataset."""
    data = TwitterTennisDatasetLoader()

    dataset = data.get_dataset()

    graph_statistics = GraphStatistics("all", to_tensors=True)(dataset)

    for sequence in graph_statistics.values():
        assert len(sequence) == len(data.features)
        for v in sequence:
            assert isinstance(v, (torch.Tensor)) or v is None


def test_statistics_work_on_empty_graph() -> None:
    """Tests edge case for graph statistics."""

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3, 4, 5])

    graph_statistics = GraphStatistics("all", to_tensors=True)([graph])

    for sequence in graph_statistics.values():
        for v in sequence:
            assert isinstance(v, (torch.Tensor)) or v is None

def test_normalize_hist_tensor()->None:
    """Tests tensor input
    """
    p = torch.tensor([1.0, 2.0, 3.0])
    normalized_p = normalize_hist(p)
    assert torch.allclose(normalized_p.sum(), torch.tensor(1.0))

def test_normalize_hist_ndarray()->None:
    """Tests ndarray input
    """
    p = np.array([1.0, 2.0, 3.0])
    normalized_p = normalize_hist(p)
    assert np.isclose(np.sum(normalized_p), 1.0)
