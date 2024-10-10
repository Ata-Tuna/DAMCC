
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

from brainiac_temporal.data.peturb_topology import AddEdges, RemoveEdges, perturb_tgt_graph_topology,_sample_n_elements, perturb_topology
from brainiac_temporal.data.datasets import fetch_insecta_dataset
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import torch


def test_perturb_topology()->None:
    """Tests the pertubation function on Data object
    """
    sample_data = fetch_insecta_dataset(colony=6)[0]
    perturbed_data = perturb_topology(sample_data, p=0.1)

    assert  not torch.equal(perturbed_data.edge_index ,sample_data.edge_index)

def test_pertubation_on_real_dataset()->None:
    """Tests perturbation on real dataset
    """
    reference_dataset = fetch_insecta_dataset(colony=6)
    pertubed_graphs = perturb_tgt_graph_topology(reference_dataset, [0.001, 0.01, 0.1, 0.5,1])
    for _, g in pertubed_graphs.items():
        assert isinstance(g ,DynamicGraphTemporalSignal)


def test_sample_n_elements()->None:
    """Tests the sampling function
    """
    n = 10
    prior = 0.5
    n_members = 5

    idx_in, idx_out = _sample_n_elements(n, prior, n_members)

    assert len(idx_in) == n_members
    assert len(idx_out) == n - n_members


def test_add_edges()->None:
    """Tests adding the edges class
    """
    sample_data = fetch_insecta_dataset(colony=6)[0]
    transform = AddEdges(p_edges=0.5)
    transformed_data = transform(sample_data)

    assert transformed_data.edge_index.shape[1] > sample_data.edge_index.shape[1]

def test_remove_edges()->None:
    """Tests removing the edges class
    """
    sample_data = fetch_insecta_dataset(colony=6)[0]
    transform = RemoveEdges(p_edges=0.5)
    transformed_data = transform(sample_data)

    assert transformed_data.edge_index.shape[1] < sample_data.edge_index.shape[1]
