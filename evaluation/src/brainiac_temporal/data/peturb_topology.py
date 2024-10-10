
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

__all__ = ["AddEdges", "RemoveEdges", "perturb_topology", "perturb_tgt_graph_topology"]

import typing as ty
import torch_geometric.transforms as tfs
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected
import torch
from torch import Tensor
from loguru import logger
from torch_geometric_temporal import DynamicGraphTemporalSignal
from typing import Dict
import numpy as np

def _sample_n_elements(
    n: int,
    prior: float = 0.5,
    n_members: int = None,
) -> ty.Tuple[torch.LongTensor, torch.LongTensor]:
    """Sample `n_members` members from a population of `n` samples.
    Args:
        n (int): Total population count.
        prior (float, optional): Prior for sampling members. Ignored if `n_members` is provided.
        n_members (int, optional): Number of members to sample. If not provided, this will be `int(n * prior)`.
    Returns:
        ty.Tuple[torch.LongTensor, torch.LongTensor]: _description_
    """
    n_members = max(1, int(n * prior)) if n_members is None else n_members
    idx = torch.randperm(n)
    idx_in: torch.LongTensor = idx[:n_members].long()  # type: ignore
    idx_out: torch.LongTensor = idx[n_members:].long()  # type: ignore

    return idx_in, idx_out


class AddEdges(tfs.BaseTransform):
    """Adds edges from the negative sampling."""

    def __init__(
        self,
        n_edges: int = None,
        p_edges: float = 0.01,
        edge_index: Tensor = None,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            n_edges (int, optional):
                Number of edges to be added. Defaults to None.
            p_edges (float, optional):
                Percentage of edges to be added. Defaults to None.
            edge_index (Tensor, optional):
                Here you can pass the edge index of the original graph. This is useful when this transform is chained after a :class:`RemoveEdges` one. This way, you do not risk to add back edges that the :class:`RemoveEdges` has removed.
        """
        super().__init__(**kwargs)
        self.p_edges = p_edges
        self.n_edges = n_edges
        # Store a reference if necessary
        self.edge_index = edge_index

    def __call__(self, data: Data) -> Data:
        """Transformation happens here."""
        data = data.clone()

        if self.edge_index is not None:
            edge_index = self.edge_index.clone().long()
        else:
            edge_index = data.edge_index.clone().long()
        edge_index = edge_index[:, edge_index[0] <= edge_index[1]].long()

        # Get number of edges to add
        if self.n_edges is not None:
            num_neg_samples = self.n_edges
        else:
            n_edges_total = int(edge_index.size(1))
            num_neg_samples = int(self.p_edges * n_edges_total)
        # Select edges to add
        negative_edges = negative_sampling(
            edge_index.long(),
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg_samples,
            force_undirected=False,
        )

        # Check if negative_edges is not empty before concatenating
        if negative_edges.numel() > 0:
            # Convert negative_edges to integer type
            negative_edges = negative_edges.to(torch.int)
            # Concatenate the tensors
            edge_index = torch.cat((edge_index, negative_edges), dim=1)
        if not data.is_directed():
            try:
                edge_index = to_undirected(
                    edge_index
                )  # pylint: disable=no-value-for-parameter
            except RuntimeError as exc:
                logger.warning(exc)
        data.edge_index = edge_index
        return data


class RemoveEdges(tfs.BaseTransform):
    """Transforms a graph by removing edges."""

    def __init__(
        self,
        n_edges: int = None,
        p_edges: float = 0.01,
        edges_idx: ty.List[int] = None,
        edge_index: Tensor = None,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            n_edges (int, optional):
                Number of edges to be removed. Defaults to None.
            p_edges (float, optional):
                Percentage of edges to be removed. Defaults to None.
            edges_idx (ty.List[int], optional):
                ID of edges to be removed. Defaults to None.
            edge_index (Tensor, optional):
                Here you can pass the edge index of the original graph. This is useful when this transform is chained after a :class:`AddEdges` one. This way, you do not risk to remove edges that the :class:`AddEdges` has added.
        """
        super().__init__(**kwargs)
        self.p_edges = p_edges
        self.n_edges = n_edges
        self.edges_idx = edges_idx
        # Store a reference if necessary
        self.edge_index = edge_index

    def __call__(self, data: Data) -> Data:
        """Transformation happens here."""
        data = data.clone()
        # Sanity checks
        if self.edge_index is not None:
            edge_index = self.edge_index.clone()
        else:
            edge_index = data.edge_index
        edge_index = edge_index[:, edge_index[0] <= edge_index[1]]
        n_edges_total = int(edge_index.size(1))
        # Sample edges to remove
        if self.edges_idx is not None:
            sampled_edges = torch.Tensor(self.edges_idx).long()
        elif self.n_edges is not None:
            sampled_edges, _ = _sample_n_elements(n_edges_total, n_members=self.n_edges)
        elif self.p_edges is not None:
            sampled_edges, _ = _sample_n_elements(n_edges_total, prior=self.p_edges)
        else:
            raise RuntimeError(
                "Please provide `p_edges`, `n_edges` or `edges_idx` when creating an instance of this class."
            )
        # Remove edges
        edge_index = self._rm_edges(edge_index, sampled_edges)

        if not data.is_directed():
            try:
                edge_index = to_undirected(
                    edge_index
                )  # pylint: disable=no-value-for-parameter
            except RuntimeError as exc:
                logger.warning(exc)
        data.edge_index = edge_index
        return data

    def _rm_edges(
        self, edge_index: torch.Tensor, edges_idx: torch.Tensor
    ) -> torch.Tensor:
        """Removes edges."""
        n_edges_total = int(edge_index.size(1))
        mask = torch.ones(n_edges_total).bool().to(edge_index.device)
        edges_idx = edges_idx.view(-1).long()
        for i in edges_idx:
            mask[i] = False
        edge_index = edge_index[:, mask]
        return edge_index


def perturb_topology(graph: Data, p: float) -> Data:
    """Perturb topology.
    Args:
        graph (Data): _description_
        p (float): _description_
    """
    perturbed_graph: Data = graph.clone()
    edges = graph.edge_index
    added_edges_graph= AddEdges(p_edges=p, edge_index=edges)(graph)
    removed_edges_graph= RemoveEdges(p_edges=p, edge_index= edges)(graph)
    edges_to_remove = removed_edges_graph.edge_index
    edges_to_add= added_edges_graph.edge_index
    # Remove edges
    mask = torch.ones(perturbed_graph.edge_index.shape[1], dtype=torch.bool)
    for edge in edges_to_remove.t():
        mask = mask & ~torch.all(perturbed_graph.edge_index == edge[:, None], dim=0)
    perturbed_graph.edge_index = perturbed_graph.edge_index[:, mask]
    # Add edges
    perturbed_graph.edge_index = torch.cat([perturbed_graph.edge_index, edges_to_add], dim=1)
    """perturb_topology = tfs.Compose(
        [
            RemoveEdges(p_edges=p, edge_index=edge_index),
            AddEdges(p_edges=p, edge_index=edge_index),
        ]
    )"""
    return perturbed_graph

def perturb_tgt_graph_topology(
    snapshots: DynamicGraphTemporalSignal,
    perturbation_ratio: ty.Sequence[float] = [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75],
) -> Dict[float, DynamicGraphTemporalSignal]:
    """_summary_

    Args:
        snapshots (DynamicGraphTemporalSignal): _description_
        perturbation_ratio (ty.Sequence[float], optional): _description_. Defaults to [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75].

    Returns:
        Dict[float, DynamicGraphTemporalSignal]: _description_
    """

    pertubed_graphs = {}
    for p in perturbation_ratio:
        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        for graph in snapshots:
            perturbed_graph = perturb_topology(graph, p)
            if "x" in perturbed_graph:
                features.append(perturbed_graph.x.numpy())
            # For graphs wihtout features
            else:
                perturbed_graph.x = np.zeros(perturbed_graph.num_nodes)
            if "y" in perturbed_graph:
                targets.append(perturbed_graph.y.numpy())
            # For graphs wihtout targets
            else:
                perturbed_graph.y = np.zeros(perturbed_graph.num_nodes)
            edge_indices.append(perturbed_graph.edge_index.numpy())
            edge_weights.append(np.ones(perturbed_graph.edge_index[0].shape[0]))

        pertubed_graphs[p] = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
        )
    return pertubed_graphs
