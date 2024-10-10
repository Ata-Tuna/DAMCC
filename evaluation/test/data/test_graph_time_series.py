
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

import pytest
from brainiac_temporal.data.graph_timeseries import GraphTimeseriesDataset  # Replace 'your_module' with the actual module name
import torch
from torch_geometric.data import Data
import typing as ty

# Define some sample data
sample_original_data = [Data(x=torch.rand(10, 5), edge_index=torch.tensor([[0, 1], [1, 2]])), Data(x=torch.rand(8, 5), edge_index=torch.tensor([[0, 1], [1, 2]]))]
sample_fake_data = [Data(x=torch.rand(10, 5), edge_index=torch.tensor([[0, 1], [1, 2]])), Data(x=torch.rand(8, 5), edge_index=torch.tensor([[0, 1], [1, 2]]))]

def test_graph_timeseries_dataset()-> None:
    """Tests for GraphTimeseriesDataset class
    """
    # Create an instance of GraphTimeseriesDataset
    dataset = GraphTimeseriesDataset(original=sample_original_data, fake=sample_fake_data)

    # Test the length of the dataset
    assert len(dataset) == len(sample_original_data) + len(sample_fake_data)

    # Test the get method for original data
    for i in range(len(sample_original_data)):
        data, label = dataset.get(i)
        assert isinstance(data, Data)
        assert label.item() == 1

    # Test the get method for fake data
    for i in range(len(sample_original_data), len(dataset)):
        data, label = dataset.get(i)
        assert isinstance(data, Data)
        assert label.item() == 1
