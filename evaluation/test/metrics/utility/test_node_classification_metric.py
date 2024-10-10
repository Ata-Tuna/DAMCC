
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

from brainiac_temporal.metrics.utility import node_classification_metric


def test_node_classification_metric() -> None:
    """Tests the node_classification_metric function.

    This metric is the ROC-AUC of a node classification task trained on a generated graph and tested on an original graph. It is bounded between 0 and 1 and higher is better.
    """
    dataset_tgt = tgt.TwitterTennisDatasetLoader().get_dataset()

    for i, batch in enumerate(dataset_tgt.targets):
        dataset_tgt.targets[i] = np.ones(batch.shape[0])
        dataset_tgt.targets[i][0] = 0

    generated = dataset_tgt
    original = dataset_tgt

    metric = node_classification_metric(
        original,
        generated,
        max_epochs=5,
    )

    assert metric > 0.5
