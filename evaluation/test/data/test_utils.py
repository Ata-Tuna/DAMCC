
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

from brainiac_temporal.data.utils import remove_isolated_nodes, perturb_featured_dataset
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import torch
def test_remove_isolated_nodes()->None:
    """Remove isolated nides over all snapshots
    """
    # Create a sample DynamicGraphTemporalSignal for testing
    edge_indices = [ ([[0, 1, 2], [1, 2, 3]]),
                    np.array([[0, 1], [1, 2]]),
                    np.array([[2, 3], [3, 4]])]

    edge_weights = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    features = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    targets = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    dynamic_signal = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets,
    )
    expected_num_nodes = 5
    # Call the function to remove isolated nodes
    transformed_signal = remove_isolated_nodes(dynamic_signal)
    for snaphsot in transformed_signal:
        assert snaphsot.x.size(0) == expected_num_nodes
        assert snaphsot.y.size(0) == expected_num_nodes




def test_perturb_featured_dataset()->None:
    """_summary_
    """
    torch.manual_seed(42)
    np.random.seed(42)
    edge_indices = [ ([[0, 1, 2], [1, 2, 3]]),
                    np.array([[0, 1], [1, 2]]),
                    np.array([[2, 3], [3, 4]])]

    edge_weights = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    features = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    targets = [np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)]),
                np.array([[1] for _ in range(10)])]

    sample_dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets,
    )
    copy_dataset= sample_dataset
    perturbed_dataset = perturb_featured_dataset(
        dataset=copy_dataset,
        features_perturbation_ratio=0.5,
        noise_scale=0.2,
        is_binary=False,
        topology_perturbation_ratio=[0.5],
    )

    # Validate the shapes and types
    assert isinstance(perturbed_dataset, DynamicGraphTemporalSignal)

    assert not np.array_equal(perturbed_dataset.features, sample_dataset.features)

    # Validate perturbed topology (edge indices)
    assert not np.array_equal(perturbed_dataset.edge_indices, sample_dataset.edge_indices)
