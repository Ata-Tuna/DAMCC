
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

from __future__ import division

from typing import Iterable, Optional, Tuple

import numpy as np
from numpy import random
from scipy.stats import kstwobign, pearsonr

# Heavily inspired from : https://github.com/syrte/ndtest/blob/master/ndtest.py

__all__ = ["statistic_quartiles_ks2d"]


def ks2d(x: np.ndarray, y: np.ndarray, nboot: Optional[int] = None) -> float:
    """Two-dimensional Kolmogorov-Smirnov test on two samples.
    Parameters
    ----------
    x : ndarray, shape (n1, 2)
        Data of sample 1.
    y : ndarray, shape (n2, 2)
        Data of sample 2. Size of two samples can be different.
    nboot : None or int
        Number of bootstrap resample to estimate the p-value. A large number is expected.
        If None, an approximate analytic estimate will be used.

    Returns
    -------
    p : float
        Two-tailed p-value.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different.
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation
    is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate,
    but it certainly implies that the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, MNRAS, 202, 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, MNRAS, 225, 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8

    """
    assert (x.ndim == y.ndim == 2) and (x.shape[1] == y.shape[1] == 2)

    x1, x2 = x[:, 0], x[:, 1]
    y1, y2 = y[:, 0], y[:, 1]
    nx, ny = len(x1), len(y1)
    D = avgmaxdist(x1, x2, y1, y2)

    if nboot is None:
        sqen = np.sqrt(nx * ny / (nx + ny))
        r1 = pearsonr(x1, x2)[0]
        r2 = pearsonr(y1, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = nx + ny
        dim1 = np.concatenate([x1, y1])
        dim2 = np.concatenate([x2, y2])
        d = np.empty(nboot, "f")
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, iy1 = idx[:nx], idx[ny:]
            d[i] = avgmaxdist(dim1[ix1], dim2[ix1], dim1[iy1], dim2[iy1])
        p = np.sum(d > D).astype("f") / nboot

    return float(p)


def avgmaxdist(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """Average of maximum distance, eq. 3.6 of Fasano & Franceschini (1987)"""
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """Maximum distance, eq. 3.5 of Fasano & Franceschini (1987)"""
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return float(max(dmin, dmax))


def quadct(
    x: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray
) -> Tuple[float, float, float, float]:
    """Equation 3.4 of Fasano & Franceschini (1987)"""
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = float(np.sum(ix1 & ix2) / n)
    b = float(np.sum(ix1 & ~ix2) / n)
    c = float(np.sum(~ix1 & ix2) / n)
    d = 1 - a - b - c
    return a, b, c, d


def statistic_quartiles_ks2d(
    original_statistics: Iterable[np.ndarray],
    generated_statistics: Iterable[np.ndarray],
    bootstrap: Optional[int] = None,
) -> float:
    """Computes the Kolmogorov-Smirnov statistic between two sets of statistics.

    Args:
        original_statistics (Iterable[np.ndarray]): The original statistics.
        generated_statistics (Iterable[np.ndarray]): The generated statistics.
        bootstrap (Optional[int], optional): Number of bootstrap resample to estimate the p-value. A large number is expected. If None, an approximate analytic estimate will be used.

    Returns:
        float: The Kolmogorov-Smirnov statistic.
    """
    original_quartiles = np.array(
        [
            np.percentile(original_statistic, [25, 75])
            for original_statistic in original_statistics
        ]
    )
    generated_quartiles = np.array(
        [
            np.percentile(generated_statistic, [25, 75])
            for generated_statistic in generated_statistics
        ]
    )

    return ks2d(original_quartiles, generated_quartiles, nboot=bootstrap)
