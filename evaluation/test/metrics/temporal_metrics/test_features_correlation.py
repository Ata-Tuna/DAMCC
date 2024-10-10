
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
import torch_geometric_temporal as tgt

from brainiac_temporal.metrics.temporal_metrics import features_correlation


def test_features_correlation_metric() -> None:
    """Tests the features_correlation_metric function."""
    dataset_tgt = tgt.EnglandCovidDatasetLoader().get_dataset()

    for i, batch in enumerate(dataset_tgt.targets):
        dataset_tgt.targets[i] = np.ones(batch.shape[0])
        dataset_tgt.targets[i][0] = 0

    generated = dataset_tgt
    original = dataset_tgt

    metric = features_correlation(original, generated)

    assert metric == 1
