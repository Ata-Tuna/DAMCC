
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
    "transform_line",
    "add_opposite_edges",
    "convert_insecta_file_to_tglib_file",
    "remove_isolated_nodes",
    "perturb_featured_dataset",
]
import typing as ty
from torch_geometric_temporal import DynamicGraphTemporalSignal
import torch
import numpy as np
from brainiac_temporal.data.perturb_feat import perturb_continuous_features
from brainiac_temporal.data.peturb_topology import perturb_tgt_graph_topology


def transform_line(line: str) -> str:
    """transforms a single line from insecta dataset:
    deleted the third column

    Args:
        line (string): line from insecta dataset file

    Returns:
        str : new line without the third column
    """
    s_line = line.split(" ")
    return " ".join(s_line[:2] + s_line[-1:])


def add_opposite_edges(line: str) -> str:
    """transforms the edge in the opposite direction

    Args:
        line (string): line from insecta dataset file

    Returns:
        str : new ledge in the opposite direction
    """
    s_line = line.split(" ")
    return " ".join(s_line[:2][::-1] + s_line[-1:])


def convert_insecta_file_to_tglib_file(path: str, opposite_edges: bool = True) -> None:
    """converts the insecta dataset in the tglib format

    Args:
        path (string): line from insecta dataset file
        opposite_edges (bool): add the edges in the opposite direction
    """
    with open(path, "r") as file:
        string_list = file.readlines()

    # remove last column and switch column 3 with column 4
    transformed_data = list(map(transform_line, string_list))

    data = transformed_data
    if opposite_edges:
        opposite_edges_list = list(map(add_opposite_edges, string_list))
        data = data + opposite_edges_list

    new_path = path + "-tglib"
    with open(new_path, "w") as file_out:
        new_file_contents = "".join(data)
        file_out.write(new_file_contents)


def remove_isolated_nodes(
    dataset: DynamicGraphTemporalSignal,
) -> DynamicGraphTemporalSignal:
    """
    Removes nodes isolated through all dataset snapshots.

    Args:
        dataset (DynamicGraphTemporalSignal): Dynamic graph dataset.

    Returns:
        DynamicGraphTemporalSignal: Dynamic graph dataset without isolated nodes.
    """
    existing_nodes: set = set()
    for snap in dataset:
        existing_nodes = existing_nodes.union(set(snap.edge_index.numpy().flatten()))

    connected_nodes_list = sorted(list(existing_nodes))
    features = []
    targets = []
    restricted_snapshots = []
    # datasets without target or node feat
    has_labels = True

    if all(element is None for element in dataset.targets) or all(
        np.isnan(element).all() for element in dataset.targets
    ):
        has_labels = False

    has_feat = True
    if all(element is None for element in dataset.features) or all(
        np.isnan(element).all() for element in dataset.features
    ):
        has_feat = False

    for snap in dataset:
        new_snap = snap.subgraph(torch.tensor(connected_nodes_list))
        restricted_snapshots.append(new_snap)

        # Update other attributes if present
        if "x" in new_snap and has_feat:
            features.append(new_snap.x.numpy())
        # For graphs wihtout features
        else:
            features.append(np.full(len(list(connected_nodes_list)), np.nan))

        if "y" in new_snap and has_labels:
            targets.append(new_snap.y.numpy())
        # For graphs wihtout targets
        else:
            targets.append(np.full(len(list(connected_nodes_list)), np.nan))

    edge_indices = [snap.edge_index.numpy() for snap in restricted_snapshots]
    edge_weights = [snap.edge_attr.numpy() for snap in restricted_snapshots]

    restricted_signal = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets,
    )
    return restricted_signal


def perturb_featured_dataset(
    dataset: DynamicGraphTemporalSignal,
    features_perturbation_ratio: float,
    is_binary: bool,
    noise_scale: float = 0.1,
    topology_perturbation_ratio:ty.Sequence[float] = [0.5],
) -> DynamicGraphTemporalSignal:
    """_summary_

    Args:
        dataset (DynamicGraphTemporalSignal): _description_
        features_perturbation_ratio (float): Probability.
        noise_scale (float): Scale of the added noise.
        is_binary (bool): Indicates if the input features are binary or not.
        topology_perturbation_ratio (ty.Sequence[float], optional): _description_. Defaults to [0.5].

    Returns:
        DynamicGraphTemporalSignal: _description_
    """
    perturbed_feat= []
    for f in dataset.features:
        perturbed_feat. append( perturb_continuous_features(
        f, features_perturbation_ratio,is_binary, noise_scale
    ))
    perturbed_topology_dataset:DynamicGraphTemporalSignal = perturb_tgt_graph_topology(dataset, topology_perturbation_ratio)[topology_perturbation_ratio[0]]
    fully_perturbed_dataset = DynamicGraphTemporalSignal(
        perturbed_topology_dataset.edge_indices,
        perturbed_topology_dataset.edge_weights,
        perturbed_feat,
        perturbed_topology_dataset.targets,
    )
    return fully_perturbed_dataset
