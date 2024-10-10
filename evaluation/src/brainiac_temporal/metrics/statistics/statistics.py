
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

__all__ = ["GraphStatistics", "normalize_hist"]

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
import torch
from statsmodels.tsa.stattools import pacf
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from .utils import spectral_worker

GRAPH_TYPE = Union[nx.Graph, Data]


class GraphStatistics:
    """Runs popular metrics from networkx on input graph. Thus, a graph is reduced to a set of metrics, which can be either scalar values, vectors or matrices."""

    def __init__(
        self,
        metrics: Union[str, List[str]] = "all",
        to_tensors: bool = True,
        normalise: bool= False,
    ) -> None:
        """
        Args:
            graph (Data | nx.Graph): Input graph.
        """
        # inputs
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.to_tensors = to_tensors
        self.normalise = normalise

    def __call__(
        self,
        graph: Union[GRAPH_TYPE, Iterable[GRAPH_TYPE]],
    ) -> Dict[str, List[Any]]:
        """Runs the metrics."""
        if isinstance(graph, (nx.Graph, Data)):
            metrics = self.compute_graph_metrics(graph)
            return self.reshape_metrics([metrics])
        metrics_list = []
        for g in graph:
            metrics_list.append(self.compute_graph_metrics(g))

        return self.reshape_metrics(metrics_list)

    def reshape_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Reshapes the metrics to a dictionary of lists."""
        reshaped_metrics: Dict[str, List[Any]] = dict()
        for metric in metrics:
            for key, val in metric.items():
                if key not in reshaped_metrics:
                    reshaped_metrics[key] = []
                reshaped_metrics[key].append(val)
        return reshaped_metrics

    def compute_graph_metrics(
        self,
        graph: GRAPH_TYPE,
    ) -> Dict[str, Any]:
        """Computes the metrics for one graph."""
        # convert input
        graph = self.convert_graph(graph)
        # run
        output: Dict[str, Any] = dict()
        for metric in self.metrics:
            if metric.lower() in ["spectral", "all"]:
                output["spectral"] = spectral_worker(graph)
            if metric.lower() in ["degree", "all"]:
                output["degree"] = np.array([d for _, d in nx.degree(graph)])
            if metric.lower() in ["degree_centrality", "all"]:
                degree_centrality: dict = nx.degree_centrality(graph)
                output["degree_centrality"] = np.array(list(degree_centrality))
            if metric.lower() in ["clustering", "all"]:
                clustering: dict = nx.clustering(graph)
                output["clustering"] = np.array(list(clustering.values()))
            if metric.lower() in ["closeness_centrality", "all"]:
                closeness: dict = nx.closeness_centrality(graph)
                output["closeness_centrality"] = np.array(list(closeness.values()))
            if metric.lower() in ["katz_centrality", "all"]:
                try:
                    katz: dict = nx.katz_centrality(graph)
                    output["katz_centrality"] = np.array(list(katz.values()))
                except Exception:  # pylint: disable=broad-except
                    output["katz_centrality"] = None
            if metric.lower() in ["eigenvector_centrality", "all"]:
                try:
                    centrality: dict = nx.eigenvector_centrality(graph)
                    output["eigenvector_centrality"] = np.array(
                        list(centrality.values())
                    )
                except Exception:  # pylint: disable=broad-except
                    output["eigenvector_centrality"] = None
            if metric.lower() in ["avg_clust_coeff", "all"]:
                output["avg_clust_coeff"] = nx.average_clustering(graph)
            if metric.lower() in ["transitivity", "all"]:
                output["transitivity"] = nx.transitivity(graph)
            if metric.lower() in ["diameter", "all"]:
                try:
                    output["diameter"] = nx.algorithms.distance_measures.diameter(graph)
                except Exception:  # pylint: disable=broad-except
                    output["diameter"] = None
            if metric.lower() in ["average_shortest_path_length", "all"]:
                try:
                    output[
                        "average_shortest_path_length"
                    ] = nx.average_shortest_path_length(graph)
                except Exception:  # pylint: disable=broad-except
                    output["average_shortest_path_length"] = None
        if self.to_tensors:
            for key, val in output.items():
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    tensor = torch.from_numpy(val)
                    if self.normalise:
                        output[key] =  normalize_hist(tensor)  # type: ignore
                    else:
                        output[key] =  tensor  # type: ignore
                else:
                    tensor = torch.tensor(val).float().view(-1)
                    output[key] = tensor  # type: ignore
        return output

    def convert_graph(self, graph: GRAPH_TYPE) -> nx.Graph:
        """Convert input graph to the needed format."""
        if isinstance(graph, Data):
            graph = to_networkx(graph, to_undirected=True)
        assert isinstance(
            graph, nx.Graph
        ), f"Input graph should be of type {GRAPH_TYPE} but was found of type {type(graph)}."
        return graph

def normalize_hist(p: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Normalize frequencies."""
    if isinstance(p, torch.Tensor):
        p = p / p.sum()
    elif isinstance(p, np.ndarray):
        p = p / np.sum(p)
    else:
        raise TypeError("Input type not supported. Use torch.Tensor or np.ndarray.")
    return p
