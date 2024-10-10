
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

import numpy as np

from brainiac_temporal.metrics.temporal_aggregations import statistic_quartiles_ks2d


def test_statistic_quartiles_ks2d() -> None:
    """Tests the statistic_quartiles_ks2d function."""

    original_statistics = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i for i in range(8)
    ]

    # This is very different from the original as it is shifted by small amounts
    generated_statistics_1 = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 0.1 for i in range(10)
    ]

    # This is similar as the sequence starts and ends with the same values but the increments are smaller (this sequence is also 10 times longer than the original)
    generated_statistics_2 = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 0.1 for i in range(80)
    ]

    # This should be very close to 1 as we are giving the same sequence twice
    metric_identical = statistic_quartiles_ks2d(
        original_statistics, original_statistics
    )
    assert abs(metric_identical - 1) < 1e-5

    # This should detect that those are two very different sequences
    metric_different_1 = statistic_quartiles_ks2d(
        original_statistics, generated_statistics_1
    )

    assert metric_different_1 < 0.2

    # This should detect that the sequences are very similar but not identical
    metric_different_2 = statistic_quartiles_ks2d(
        original_statistics, generated_statistics_2
    )
    assert metric_different_2 > 0.8


def test_ks2d_with_bootstrap() -> None:
    """Tests that the ks2d function works with bootstrap."""
    original_statistics = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i for i in range(8)
    ]

    # This is very different from the original as it is shifted by small amounts
    generated_statistics = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 0.1 for i in range(10)
    ]

    # This should be very close to 1 as we are giving the same sequence twice
    metric = statistic_quartiles_ks2d(
        original_statistics, generated_statistics, bootstrap=1000
    )
    assert metric < 0.2
