
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

__all__ = ["MetricEvaluator"]

from typing import Any, Dict, Iterable, List, Optional, Union

import sys
import pytglib as tgl

import networkx as nx
import numpy as np
import torch
from dtaidistance.dtw import distance_fast as dtw_1d
from torch_geometric.data import Data
from brainiac_temporal.data.utils import remove_isolated_nodes
from brainiac_temporal.metrics.utils import from_tgt_to_tglib
from brainiac_temporal.metrics.statistics.statistics import GraphStatistics
from brainiac_temporal.metrics.temporal_aggregations.dtw import quantile_dtw
from brainiac_temporal.metrics.temporal_aggregations import pacf_on_metrics
from brainiac_temporal.metrics.temporal_metrics import temporal_correlation_difference
from brainiac_temporal.metrics.utility.link_forecasting import link_forecasting_metric
from brainiac_temporal.metrics.utility.node_classification import (
    node_classification_metric,
)
from brainiac_temporal.metrics.nndr import get_nndr
from brainiac_temporal.metrics.nndr import get_snapshots_embeddings, load_lp_embedder, transform_embeddings_matrix

class MetricEvaluator:
    def __init__(
        self,
        statistics: Optional[Union[str, List[str]]] = "all",
        temporal_aggregation: str = "dtw",
        temporal_aggregation_kwargs: Optional[Dict[str, Any]] = None,
        utility_metrics: Optional[Union[str, List[str]]] = "auto",
        utility_metrics_kwargs: Optional[Dict[str, Any]] = None,
        temporal_metrics: Optional[Union[str, List[str]]] = "auto",
        get_privacy_metric: bool = False,
        nndr_calculation_method: str = "dtw",
        embedder_path: str = "",
        normalise_metrics: bool = True,
    ) -> None:
        """
        This class evaluates all available metrics on a given generated and original dataset. It is used to evaluate the quality of a generative model.

        The available metrics are:
            - Temporal Aggregation (e.g. DTW on quantiles of the statistics) of graph topological statistics (e.g. degree distribution, clustering coefficient, etc.)
            - Utility metrics (e.g. ROC-AUC of a node classification task and of a link forecasting task trained on the generated graph and evaluated on the original graph)
            - Temporal metrics (e.g. temporal correlation of the node features).
        """

        if temporal_aggregation not in ["dtw", "pacf"]:
            raise ValueError(
                f"Invalid temporal aggregation method {temporal_aggregation}."
            )

        if isinstance(utility_metrics, str) and utility_metrics != "auto":
            raise ValueError(f"Invalid utility metric {utility_metrics}.")
        elif isinstance(utility_metrics, list):
            for metric in utility_metrics:
                if metric not in ["node_classification", "link_forecasting"]:
                    raise ValueError(f"Invalid utility metric {metric}.")

        self.statistics = statistics
        self.temporal_aggregation = temporal_aggregation
        self.temporal_aggregation_kwargs = temporal_aggregation_kwargs or {}
        self.utility_metrics = utility_metrics
        self.utility_metrics_kwargs = utility_metrics_kwargs or {}
        self.temporal_metrics = temporal_metrics
        self.statistic_evaluator: Optional[GraphStatistics] = None
        self.get_privacy_metric= get_privacy_metric
        self.nndr_calculation_method = nndr_calculation_method
        self.embedder_path = embedder_path
        self.normalise_metrics = normalise_metrics
        if self.statistics is not None:
            self.statistic_evaluator = GraphStatistics(self.statistics, to_tensors=True, normalise= self.normalise_metrics)

    def __call__(
        self,
        original: Iterable[Data],
        generated: Iterable[Data],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Computes the metrics on the original and generated sequences."""
        #transform graphs
        results = {}
        original, generated = remove_isolated_nodes(original), remove_isolated_nodes(generated)
        if self.statistics is not None:
            aggregated_statistics = self._compute_statistics_and_temporal_aggregation(
                original, generated
            )

            results.update(aggregated_statistics)

        if self.utility_metrics is not None:
            utility_metrics = self._compute_utility_metrics(original, generated)

            results.update(utility_metrics)

        if self.temporal_metrics is not None:
            temporal_metrics = self._compute_temporal_metrics(original, generated )

            results.update(temporal_metrics)

        if self.get_privacy_metric:
            nndr={}
            privacy_metric= self._compute_nndr(original, generated )
            nndr["NNDR_mean"] = float( np.round(np.mean(privacy_metric["nndr_score"]),3))
            nndr["NNDR_std"] =  float(np.round(np.std(privacy_metric["nndr_score"]),3))
            results.update(nndr)

        return results

    def _compute_statistics_and_temporal_aggregation(
        self,
        original: Iterable[Data],
        generated: Iterable[Data],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Computes the statistics and the temporal aggregation of the statistics."""
        if self.statistic_evaluator is None:
            raise ValueError("No statistics to compute.")
        original_statistics = self.statistic_evaluator(original)
        generated_statistics = self.statistic_evaluator(generated)
        results = {}

        for k in original_statistics.keys():
            print(k)
            results[k] = self._compute_temporal_aggregation(
                original_statistics[k], generated_statistics[k]
            )
        return results

    def _compute_temporal_aggregation(
        self,
        original: Iterable[Optional[torch.Tensor]],
        generated: Iterable[Optional[torch.Tensor]],
    ) -> Union[float, np.ndarray]:
        """Computes the temporal aggregation of the statistics."""
        if (
            len([t for t in original if t is not None]) == 0
            or len([t for t in generated if t is not None]) == 0
        ):
            return np.nan

        is_multidim = False
        for tensor in generated:
            if tensor is not None:
                is_multidim = len(tensor) > 1
                break

        if self.temporal_aggregation == "dtw":
            if is_multidim:
                pruned_original = [t.cpu().numpy() for t in original if t is not None]
                pruned_generated = [t.cpu().numpy() for t in generated if t is not None]

                return quantile_dtw(
                    pruned_original,
                    pruned_generated,
                    **self.temporal_aggregation_kwargs,
                )

            else:
                np_original = np.array([t.item() for t in original if t is not None])
                np_generated = np.array([t.item() for t in generated if t is not None])

                return float(dtw_1d(np_original, np_generated))
        elif self.temporal_aggregation == "pacf":
            if not is_multidim:
                pruned_original = [t.cpu().numpy() for t in original if t is not None]
                pruned_generated = [t.cpu().numpy() for t in generated if t is not None]

                return pacf_on_metrics(pruned_original, pruned_generated, **self.temporal_aggregation_kwargs)
            else:
                return np.nan
        else:
            return np.nan

    def _compute_utility_metrics(
        self,
        original: Iterable[Data],
        generated: Iterable[Data],
    ) -> Dict[str, float]:
        """Computes the utility metrics."""
        metrics_to_compute = []

        has_labels = True
        if all(np.isnan(element.y).all() for element in original):
            has_labels = False

        if all(np.isnan(element.y).all() for element in generated):
            has_labels = False

        if self.utility_metrics == "auto":
            print(has_labels)
            if has_labels:
                metrics_to_compute.append("node_classification")
            metrics_to_compute.append("link_forecasting")
        elif isinstance(self.utility_metrics, list):
            metrics_to_compute.extend(self.utility_metrics)

        results = {}

        if "node_classification" in metrics_to_compute:
            results["node_classification"] = node_classification_metric(
                original, generated, **self.utility_metrics_kwargs
            )

        if "link_forecasting" in metrics_to_compute:
            results["link_forecasting"] = link_forecasting_metric(
                original, generated, **self.utility_metrics_kwargs
            )

        return results

    def _compute_metric_on_iterable(
        self,
        iterable: Iterable[Data],
        metric: str
    ) -> List[float]:
        """Computes temporal metrics."""
        metric_all: List[float] = []
        for g in iterable:
            # node level metric
            if metric == 'temporal_closeness':
                metric_g = tgl.temporal_closeness(g, tgl.Distance_Type.Fastest)
            elif metric == 'temporal_clustering_coefficient':
                tg = tgl.to_incident_lists(g)
                metric_g = tgl.temporal_clustering_coefficient(tg, tg.getTimeInterval())
            else:
                metric_g = []

            metric_all = metric_all + list(metric_g)
        return metric_all

    def _compute_nndr( self,
        original: Iterable[Data],
        generated: Iterable[Data],) -> Dict[str, np.ndarray]:

        nndr={}
        embedder = load_lp_embedder(self.embedder_path, original)
        orig_embeddings = get_snapshots_embeddings(embedder, original)
        gen_embeddings = get_snapshots_embeddings(embedder,generated)
        embeddings_matrix = transform_embeddings_matrix(orig_embeddings,gen_embeddings, self.nndr_calculation_method)
        (score, hist, edges) = get_nndr(torch.Tensor(embeddings_matrix))
        nndr["nndr_score"] = score.cpu().numpy()
        nndr ["nndr_histogram"] = hist.cpu().numpy()
        nndr["nndr_edges"]= edges.cpu().numpy()
        return nndr


    def _compute_temporal_metrics(
        self,
        original: Iterable[Data],
        generated: Iterable[Data],
    ) -> Dict[str, float]:
        """Computes the temporal metrics."""
        results = {}

        if self.temporal_metrics == "auto" or (
            isinstance(self.temporal_metrics, list)
            and "temporal_correlation" in self.temporal_metrics
        ):
            results["temporal_correlation"] = temporal_correlation_difference(
                original, generated
            )

        if self.temporal_metrics == "auto" or (isinstance(self.temporal_metrics, list) and "temporal_closeness" in self.temporal_metrics):

            ### Tglib metrics
            original_tglib = from_tgt_to_tglib(original)
            generated_tglib = from_tgt_to_tglib(generated)

            ## temporal closeness metric

            # compute temporal closeness for each node in the original and generated
            closeness_original = self._compute_metric_on_iterable([original_tglib], 'temporal_closeness')
            closeness_generated = self._compute_metric_on_iterable([generated_tglib], 'temporal_closeness')

            # average
            avg_closeness_original = sum(closeness_original) / len(closeness_original)
            avg_closeness_generated = sum(closeness_generated) / len(closeness_generated)

            results["diff_avg_temporal_closeness"] = abs(avg_closeness_original-avg_closeness_generated)

        if self.temporal_metrics == "auto" or (isinstance(self.temporal_metrics, list) and "temporal_clustering_coefficient" in self.temporal_metrics):

            ### Tglib metrics
            original_tglib = from_tgt_to_tglib(original)
            generated_tglib = from_tgt_to_tglib(generated)

            ## temporal_clustering_coefficient
            clustering_original = self._compute_metric_on_iterable([original_tglib], 'temporal_clustering_coefficient')
            clustering_generated = self._compute_metric_on_iterable([generated_tglib], 'temporal_clustering_coefficient')


            # average
            avg_clustering_original = sum(clustering_original) / len(clustering_original)
            avg_clustering_generated = sum(clustering_generated) / len(clustering_generated)

            results["diff_avg_temporal_clustering_coefficient"] = abs(avg_clustering_original-avg_clustering_generated)

        return results
