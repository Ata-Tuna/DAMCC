
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
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader

from brainiac_temporal.metrics.statistics import GraphStatistics
from brainiac_temporal.metrics.temporal_aggregations import pacf_on_metrics


def test_statistics_pcaf_on_tgt() -> None:
    """Tests the PACF on metrics calculated on snapshits of a torch_geometric_temporal dataset."""
    data = TwitterTennisDatasetLoader()

    dataset = data.get_dataset()

    graph_statistics = GraphStatistics("all", to_tensors=True)(dataset)
    metric = graph_statistics["avg_clust_coeff"]
    t1 = [t.cpu().numpy() for t in metric if t is not None]
    t2 = [t.cpu().numpy() for t in metric if t is not None]
    pcaf_results = pacf_on_metrics(t1, t2, max_lag=None)
    for e in pcaf_results:
        assert isinstance(e, float)


def test_statistics_pcaf_on_repeated_seq() -> None:
    """Tests the PACF on sequence of same values."""
    x = [1, 1, 1, 1, 1, 1, 1, 1]
    y = [1, 1, 1, 1, 1, 1, 1, 1]
    max_lag = 2  # max_lag est valide
    result = pacf_on_metrics(x, y, max_lag)
    assert sum(result) == 0
