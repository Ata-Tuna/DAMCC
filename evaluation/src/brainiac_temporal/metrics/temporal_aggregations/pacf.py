
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

__all__ = ["pacf_on_metrics"]

import torch
from typing import Any, Dict, Iterable, List, Sequence, Union, Optional
from statsmodels.tsa.stattools import pacf
import numpy as np


def pacf_on_metrics(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute the Partial Autocorrelation Function (PACF) for a given metric

    Args:
        x (Iterable[np.ndarray]): The original statistics.
        y (Iterable[np.ndarray]): The generated statistics.
        max_lag (int):   Number of lags to return autocorrelation for. If not provided, uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,).

    Returns:
        np.ndarray: PACF values for the difference between x and y.
    """
    # Check if max_lag is valid
    if max_lag is not None:
        if max_lag >= np.size(x) or max_lag >= np.size(y):
            raise ValueError(
                "max_lag must be less than the length of sequences x and y."
            )

    # Compute PACf of x and y
    pacf_x = pacf(x, nlags=max_lag)
    pacf_y = pacf(y, nlags=max_lag)
    result = np.abs(pacf_y - pacf_x)

    return np.array(result)
