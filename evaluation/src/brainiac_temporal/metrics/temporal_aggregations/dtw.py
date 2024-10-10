
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

__all__ = ["quantile_dtw"]

from typing import Iterable

import numpy as np
from dtaidistance.dtw_ndim import distance_fast as dtw


def quantile_dtw(
    x: Iterable[np.ndarray],
    y: Iterable[np.ndarray],
    quantiles: Iterable[float] = [0.25, 0.75],
) -> float:
    """
    Computes the dynamic time warping distance on the sequence of quantiles of the different distributions.

    Args:
        x (Iterable[np.ndarray]): The original statistics.
        y (Iterable[np.ndarray]): The generated statistics.
        quantiles (Iterable[float], optional): The quantiles to use. Defaults to [0.25, 0.75].

    Returns:
        float: The dynamic time warping distance.
    """
    percentiles = np.array(quantiles) * 100

    x_quantiles = np.array([np.percentile(x_, percentiles) for x_ in x])
    y_quantiles = np.array([np.percentile(y_, percentiles) for y_ in y])

    return float(dtw(x_quantiles, y_quantiles))
