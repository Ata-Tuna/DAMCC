
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

__all__ = [
    "spectral_worker",
]

from typing import Union

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

GRAPH_TYPE = Union[nx.Graph, Data]


def spectral_worker(graph: GRAPH_TYPE) -> np.ndarray:
    """Spectral worker."""
    if isinstance(graph, Data):
        graph = to_networkx(graph)
    assert isinstance(graph, nx.Graph)
    graph = graph.to_undirected()
    eigs = np.array(eigvalsh(nx.normalized_laplacian_matrix(graph).todense())).astype(
        float
    )
    return eigs
