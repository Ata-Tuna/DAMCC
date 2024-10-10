
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
import torch
import numpy as np
from brainiac_temporal.data.perturb_feat import perturb_continuous_features


def test_perturb_continuous_features_numpy()->None:
    """Test perturbation on binary feat with np.ndarray type
    """

    numpy_feat= np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 0.5
    noise_scale = 0.1
    is_binary = False

    perturbed_data = perturb_continuous_features(numpy_feat, p, is_binary, noise_scale)

    assert isinstance(perturbed_data, np.ndarray)
    assert np.shape(perturbed_data) == np.shape(numpy_feat)

def test_perturb_continuous_features_tensor()->None:
    """Test perturbation on binary feat with tensor type
    """
    tensor_feat= torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = 0.5
    noise_scale = 0.1
    is_binary = False

    perturbed_data = perturb_continuous_features(tensor_feat, p, is_binary, noise_scale)

    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == tensor_feat.shape

def test_perturb_continuous_features_binary()->None:
    """Test perturbation on continious feat with np.ndarray type
    """
    numpy_feat= np.array([[0, 1, 1], [1, 0, 1]], dtype=np.float32)
    p = 0.5
    noise_scale = 0.1
    is_binary = True

    perturbed_data = perturb_continuous_features(numpy_feat, p, is_binary, noise_scale)

    assert isinstance(perturbed_data, np.ndarray)
    assert np.shape(perturbed_data) == np.shape(numpy_feat)

def test_perturb_continuous_features_tensor_binary()->None:
    """Test perturbation on continious feat with tensor type
    """
    tensor_feat = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float32)
    p = 0.5
    noise_scale = 0.1
    is_binary = True

    perturbed_data = perturb_continuous_features(tensor_feat, p, is_binary, noise_scale)

    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == tensor_feat.shape
