
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

from brainiac_temporal.metrics.nndr import get_nndr
import torch
import pytest
from loguru import logger


def test_get_nndr_function()->None:
    """Tests nndr instances
    """
     # Test case 1: Using a distance matrix (L2 method)
    distances = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    nndr, hist, edges = get_nndr(distances, method="l2", idx=1)
    assert isinstance(nndr, torch.Tensor)
    assert isinstance(hist, torch.Tensor)
    assert isinstance(edges, torch.Tensor)
    #  Using L2 method with two tensors
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    nndr, hist, edges = get_nndr(x, y, method="l2", idx=1)
    assert isinstance(nndr, torch.Tensor)
    assert isinstance(hist, torch.Tensor)
    assert isinstance(edges, torch.Tensor)

def test_nndr_two_elements() -> None:
    """Test the inner function with two elements in the input."""
    x = torch.Tensor([[0, 0], [0, 1]])
    y = torch.Tensor([[0, 0], [0, 1]])
    NNDR, _, _ = get_nndr(x, y)
    logger.info(f"NNDR: {NNDR}")
    assert NNDR.size(0) == 2
    t = 0
    val = NNDR[0]
    assert val == pytest.approx(t, abs=1e-3), f"Got {val}, expected {t}"


def test_nndr_mixed_elements() -> None:
    """Test the inner function with mixed elements in the input."""
    x = torch.Tensor([[0, 0], [0, 1], [0, 1], [2, 2]])
    y = torch.Tensor([[0, 0], [0, 0]])
    NNDR, _, _ = get_nndr(x, y)
    logger.info(f"NNDR: {NNDR}")
    assert NNDR.size(0) == 4
    t = 1
    val = NNDR[0]
    assert val == pytest.approx(t, abs=1e-3), f"Got {val}, expected {t}"
