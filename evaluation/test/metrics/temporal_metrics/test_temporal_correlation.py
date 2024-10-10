
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

import torch
from torch_geometric.data import Data

from brainiac_temporal.metrics.temporal_metrics import (
    temporal_correlation,
    temporal_correlation_difference,
)


def test_temporal_correlation_identical() -> None:
    """Tests the temporal_correlation function for a series of identical graphs."""
    graph_1 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    assert temporal_correlation([graph_1, graph_1]) == 1.0


def test_temporal_correlation_close() -> None:
    """Tests the temporal_correlation function for a series of very similar graphs."""
    graph_1 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    graph_2 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    assert temporal_correlation([graph_1, graph_2]) == 0.75


def test_temporal_correlation_different() -> None:
    """Tests the temporal_correlation function for a series of very different graphs."""
    graph_1 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    graph_2 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[2, 3, 3, 1, 0], [0, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    assert temporal_correlation([graph_1, graph_2]) == 0.0


def test_temporal_correlation_difference() -> None:
    """Tests the temporal_correlation_difference function."""
    graph_1 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    graph_2 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    graph_3 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[2, 3, 3, 1, 0], [0, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    assert (
        temporal_correlation_difference([graph_1, graph_2], [graph_1, graph_3]) == 0.75
    )


def test_with_isolated_node() -> None:
    """Tests the temporal_correlation function with an isolated node."""
    graph_1 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 3]]),
        num_nodes=4,
    )

    graph_2 = Data(
        x=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0], [1, 0, 2, 1, 2]]),
        num_nodes=4,
    )

    assert temporal_correlation([graph_1, graph_2]) > 0.5
