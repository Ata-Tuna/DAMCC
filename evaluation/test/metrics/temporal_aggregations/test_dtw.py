
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

from copy import deepcopy
from random import shuffle

import numpy as np

from brainiac_temporal.metrics.temporal_aggregations import quantile_dtw


def test_quantile_dtw() -> None:
    """Tests the quantile_dtw function."""

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
    metric_identical = quantile_dtw(original_statistics, original_statistics)

    assert abs(metric_identical) < 1e-5

    # This should detect that those are two very different sequences
    metric_different_1 = quantile_dtw(original_statistics, generated_statistics_1)

    assert metric_different_1 > 1

    # This should detect that the sequences are very similar but not identical
    metric_different_2 = quantile_dtw(original_statistics, generated_statistics_2)

    assert metric_identical < metric_different_2 < metric_different_1


def test_suffled_sequence_gets_bad_score() -> None:
    """Tests that a shuffled sequence gets a bad score."""
    original_statistics = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i for i in range(8)
    ]

    generated_statistics = deepcopy(original_statistics)
    shuffle(generated_statistics)

    # This should detect that those are two very different sequences
    metric = quantile_dtw(original_statistics, generated_statistics)

    assert metric > 1
