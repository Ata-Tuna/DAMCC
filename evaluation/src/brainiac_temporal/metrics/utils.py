
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
    "from_tgt_to_tglib",
]

from typing import List, Union

import sys, os

import pytglib as tgl
import torch_geometric_temporal as tgt

STATIC_GRAPH = tgt.signal.static_graph_temporal_signal.StaticGraphTemporalSignal
DYNAMIC_GRAPH = tgt.signal.dynamic_graph_temporal_signal.DynamicGraphTemporalSignal


def from_tgt_to_tglib(
    graph: Union[STATIC_GRAPH, DYNAMIC_GRAPH],  # type: ignore
) -> tgl.OrderedEdgeList:
    """Transform a tgt graph into tglib graph."""

    temporal_graph = None
    lines = []

    if isinstance(graph, STATIC_GRAPH):
        edges = graph.edge_index
        for e in edges.T:
            x = " ".join(map(str, e.tolist() + ["0\n"]))
            lines.append(x)
    else:
        for t, snapshot in enumerate(graph):  # type: ignore
            edges = snapshot.edge_index
            for e in edges.T:
                x = " ".join(map(str, e.tolist() + [f"{t}\n"]))
                lines.append(x)

    with open("temporary_file.txt", "a") as f:
        f.writelines(lines)
    temporal_graph = tgl.load_ordered_edge_list("temporary_file.txt")
    os.remove("temporary_file.txt")

    return temporal_graph
