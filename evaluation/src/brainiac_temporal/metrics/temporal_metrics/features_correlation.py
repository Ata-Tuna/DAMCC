
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

__all__ = ["features_correlation"]

import numpy as np
from torch_geometric_temporal import DynamicGraphTemporalSignal  # type: ignore


def features_correlation(
    original: DynamicGraphTemporalSignal,
    generated: DynamicGraphTemporalSignal,
) -> float:
    """
    Calculate features correlation between two DynamicGraphTemporalSignal objects.

    Args:
        original (DynamicGraphTemporalSignal): The original graph.
        generated (DynamicGraphTemporalSignal): The generated graph.

    Returns:
        float: The features correlation coefficient.
    """
    # Extract the temporal signals from the objects
    signal_original = original.features
    signal_generated = generated.features

    # Calculate the mean of the original signal
    mean_original = np.mean(signal_original, axis=0)

    # Calculate the mean of the generated signal
    mean_generated = np.mean(signal_generated, axis=0)

    # Calculate the numerator (covariance) and denominators (variances)
    numerator = np.sum(
        (signal_original - mean_original) * (signal_generated - mean_generated), axis=0
    )
    denominator_original = np.sum((signal_original - mean_original) ** 2, axis=0)
    denominator_generated = np.sum((signal_generated - mean_generated) ** 2, axis=0)

    # Calculate the correlation coefficients for each node
    correlations = numerator / np.sqrt(denominator_original * denominator_generated)

    # Calculate the mean correlation coefficient
    mean_correlation = np.mean(correlations)

    return float(mean_correlation)
