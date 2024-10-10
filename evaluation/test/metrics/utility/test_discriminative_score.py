
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

import pytest, sys, typing as ty
from loguru import logger
import numpy as np
import networkx as nx
import torch
import torch_geometric_temporal as tgt


from brainiac_temporal.types import GRAPH_SERIES
from brainiac_temporal.metrics.utility import discriminative_score_metric


def test_discriminative_score_metric() -> None:
    """Tests the discriminative score metric function."""
    # Params
    node_features = 100
    node_count = 1000
    sequence_len = 4
    edge_per_node = 15
    epochs = 2
    learning_rate = 0.01
    weight_decay = 5e-4

    # Generate original graphs
    data_original: ty.List[GRAPH_SERIES] = []
    for _ in range(25):
        snapshots = generate_temporal_graph(
            sequence_len,
            node_count,
            edge_per_node,
            node_features,
        )
        data_original.append(snapshots)

    # Generate fake graphs
    edge_per_node = 3
    data_generated: ty.List[GRAPH_SERIES] = []
    for _ in range(25):
        snapshots = generate_temporal_graph(
            sequence_len,
            node_count,
            edge_per_node,
            node_features,
        )
        data_generated.append(snapshots)

    # Run metric
    metric = discriminative_score_metric(
        data_original,  # type: ignore
        data_generated,  # type: ignore
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Test
    logger.info(f"Metric: {metric}")
    assert metric < 0.1


def generate_temporal_graph(
    sequence_len: int,
    node_count: int,
    edge_per_node: int,
    node_features: int,
) -> GRAPH_SERIES:
    """Generates temporal graph snapshots

    Args:
        sequence_len (int): the length of the graph time series.
        node_count (int): the number of nodes.
        edge_per_node (int): the number of edges per node.
        node_features (int): the number of features.

    Returns:
        ty.List[ty.Tuple[FloatTensor, LongTensor, FloatTensor]] : the list of snapshots
    """
    snapshots = []
    for _ in range(sequence_len):
        graph = nx.watts_strogatz_graph(node_count, edge_per_node, 0.5)
        edge_index = torch.LongTensor(np.array(graph.edges()).T)
        X = torch.FloatTensor(np.random.uniform(-1, 1, (node_count, node_features)))
        edge_weight = torch.FloatTensor(np.random.uniform(0, 1, (edge_index.shape[1])))
        snapshots.append((X, edge_index, edge_weight))
    return snapshots


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
