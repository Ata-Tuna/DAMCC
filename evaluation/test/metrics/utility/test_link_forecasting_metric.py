
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

import torch_geometric_temporal as tgt

from brainiac_temporal.metrics.utility import link_forecasting_metric


def test_link_forecasting_metric() -> None:
    """Tests the link_forecasting_metric function.

    This metric is the ROC-AUC of a link forecasting task trained on a generated graph and tested on an original graph.

    It is bounded between 0 and 1 and higher is better.
    """
    dataset_tgt = tgt.TwitterTennisDatasetLoader().get_dataset()

    generated = dataset_tgt
    original = dataset_tgt

    metric = link_forecasting_metric(
        original,
        generated,
        max_epochs=5,
        link_predictor_mlp_hidden_sizes=[32],
        link_predictor_embedding_size=32,
    )

    assert metric > 0.5
