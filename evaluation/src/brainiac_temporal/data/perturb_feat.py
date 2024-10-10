
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


_all__ = ["perturb_continuous_features"]

import numpy as np
import typing as ty
import torch


def perturb_continuous_features(
    features: ty.Union[np.ndarray, torch.Tensor],
    p: float,
    is_binary: bool,
    noise_scale: float = 0.1,
) -> ty.Union[np.ndarray, torch.Tensor]:
    """_summary_

    Args:
        features (ty.Union[np.ndarray, torch.Tensor]): Input features
        p (float): Probability.
        noise_scale (float): Scale of the added noise.
        is_binary (bool): Indicates if the input features are binary or not.

    Returns:
        ty.Union[np.ndarray, torch.Tensor]: Perturbed features
    """

    if isinstance(features, np.ndarray):
        feat:torch.Tensor = torch.from_numpy(features)
    else:
        feat= features.clone()
    # If the features are binary, perturb them like in SANGEA: https://code.euranova.eu/rd/bishop/brainiac-2-privacy/-/blob/develop/src/brainiac_2/utils/graphs.py?ref_type=heads
    if is_binary:
        rands = torch.rand_like(feat)
        index = rands < p
        feat[index] = 1 - feat[index]
    else:
        # Convert feat to float before adding noise
        feat = feat.float()
        # Add noise with a specified scale
        rands = torch.rand(*feat.shape)
        index = rands < p
        feat[index] += torch.randn(*feat[index].shape) * noise_scale

    if isinstance(features, np.ndarray):
        np_feat:np.ndarray = feat.cpu().numpy()
        return np_feat
    #if the input feat are tensors
    return feat
