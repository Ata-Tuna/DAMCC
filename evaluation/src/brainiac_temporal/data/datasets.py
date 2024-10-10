
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

__all__ = ["fetch_insecta_dataset", "load_imdb_dynamic_tgt", "load_tigger_datasets_tgt"]

import os
import tempfile
import urllib.request
import zipfile
from typing import Optional
import re
import numpy as np
import pandas as pd
from torch_geometric_temporal import (
    DynamicGraphTemporalSignal,
    EnglandCovidDatasetLoader,
    TwitterTennisDatasetLoader,
)
import networkx as nx


def fetch_insecta_dataset(colony: Optional[int] = None) -> DynamicGraphTemporalSignal:
    """
    Download and get the Insecta dataset in a DynamicGraphTemporalSignal object.

    Args:
        colony (Optional[int]): Colony number to download. If None, all colonies are combined. Defaults to None.
    """

    max_index = 0
    max_timestamp = -1
    colony_edge_indices = []
    colony_edge_weights = []

    colonies_to_combine = [colony] if colony is not None else range(1, 7)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for colony in colonies_to_combine:
            url = f"https://nrvis.com/download/data/dynamic/insecta-ant-colony{colony}.zip"

            urllib.request.urlretrieve(
                url, os.path.join(tmpdirname, f"insecta-ant-colony{colony}.zip")
            )

            # unzip it
            with zipfile.ZipFile(
                os.path.join(tmpdirname, f"insecta-ant-colony{colony}.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Load the file
            colony_df = pd.read_csv(
                os.path.join(tmpdirname, f"insecta-ant-colony{colony}.edges"),
                header=None,
                sep=" ",
            )

            edge_indices_per_timestamp = []
            edge_weights_per_timestamp = []

            for timestamp in sorted(np.unique(colony_df[3])):
                data = colony_df[colony_df[3] == timestamp].drop(columns=[3]).to_numpy()
                edge_index = data[:, :2].T
                edge_weight = data[:, 2]

                # double edges other way around
                edge_index = (
                    np.concatenate((edge_index, edge_index[[1, 0], :]), axis=1) - 1
                )
                edge_weight = np.concatenate((edge_weight, edge_weight))

                edge_index += max_index

                edge_indices_per_timestamp.append(edge_index)
                edge_weights_per_timestamp.append(edge_weight)

            max_timestamp = max(max_timestamp, len(edge_indices_per_timestamp))
            max_index = np.max([np.max(ei) for ei in edge_indices_per_timestamp]) + 1

            colony_edge_indices.append(edge_indices_per_timestamp)
            colony_edge_weights.append(edge_weights_per_timestamp)

        # Combine all colonies into one graph
        combined_edge_indices = []
        combined_edge_weights = []
        for timestamp in range(max_timestamp):
            edge_index = []
            edge_weight = []
            for colony in range(len(colony_edge_indices)):
                if timestamp < len(colony_edge_indices[colony]):
                    edge_index.append(colony_edge_indices[colony][timestamp])
                    edge_weight.append(colony_edge_weights[colony][timestamp])
            edge_index = np.concatenate(edge_index, axis=1)
            edge_weight = np.concatenate(edge_weight)
            combined_edge_indices.append(edge_index)
            combined_edge_weights.append(edge_weight)

        features = []
        targets = []
        for timestamp in range(max_timestamp):
            features.append(np.ones((max_index + 1, 1)))
            targets.append(np.ones(max_index + 1))

        # Regroup into dynamic graph object
        dynamic_graph = DynamicGraphTemporalSignal(
            edge_indices=combined_edge_indices,
            edge_weights=combined_edge_weights,
            features=features,
            targets=targets,
        )

        return dynamic_graph


def load_imdb_dynamic_tgt() -> DynamicGraphTemporalSignal:
    """
    Download and get the Imdb dynamic dataset in a DynamicGraphTemporalSignal object.

    """

    imdb_edge_indices = []
    imdb_edge_weights = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        url = f"https://nrvis.com/download/data/dynamic/imdb.zip"

        urllib.request.urlretrieve(url, os.path.join(tmpdirname, f"imdb.zip"))

        # unzip it
        with zipfile.ZipFile(os.path.join(tmpdirname, f"imdb.zip"), "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        # Load the file
        imdb_df = pd.read_csv(
            os.path.join(tmpdirname, f"imdb.edges"),
            header=None,
            sep=",",
        )

        for timestamp in sorted(np.unique(imdb_df[3])):
            data = imdb_df[imdb_df[3] == timestamp].drop(columns=[3]).to_numpy()
            edge_index = data[:, :2].T
            edge_weight = data[:, 2]

            # double edges other way around
            edge_index = np.concatenate((edge_index, edge_index[[1, 0], :]), axis=1) - 1
            edge_weight = np.concatenate((edge_weight, edge_weight))

            imdb_edge_indices.append(edge_index)
            imdb_edge_weights.append(edge_weight)

        features = []
        targets = []
        # we want to get the max number of noves over all snapshots
        nb_nodes = 0
        for edges in imdb_edge_indices:
            nb_nodes = max(edges.max(), nb_nodes) + 1

        for _ in range(len(imdb_edge_weights)):
            features.append(np.ones(nb_nodes).reshape([nb_nodes, 1]))
            targets.append(np.ones(nb_nodes))

        # Regroup into dynamic graph object
        dynamic_graph = DynamicGraphTemporalSignal(
            edge_indices=imdb_edge_indices,
            edge_weights=imdb_edge_weights,
            features=features,
            targets=targets,
        )

        return dynamic_graph


def load_tigger_datasets_tgt(
    dataset_name: str, is_undirected: bool = False
) -> DynamicGraphTemporalSignal:
    """Function to load datasets from tigger repository https://github.com/data-iitd/tigger/tree/main/data

    Args:
        dataset_name (str): Name of the dataset
        is_undirected (bool): if the graph is undirected then the edges are counted x2

    Raises:
        ValueError: if the dataset is unknwon this error is returned

    Returns:
        DynamicGraphTemporalSignal: tgt graph dataset version of tigger datasets
    """

    edge_indices = []
    edge_weights = []
    nb_nodes = 0
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Start with loading datasets from repository
        if dataset_name.lower() == "bitcoin":
            url = f"https://raw.githubusercontent.com/data-iitd/tigger/main/data/bitcoin/data.csv"
            # Load the CSV file into a DataFrame
            df = pd.read_csv(url, sep=",", header=None)
            df = df.iloc[:, 1:]

        elif dataset_name.lower() == "reddit":
            url = "https://raw.githubusercontent.com/data-iitd/tigger/main/data/CAW_data/reddit_processed.csv"
            # Load the CSV file into a DataFrame
            df = pd.read_csv(url, sep=",", header=None)

        elif (
            re.sub(r"[^a-zA-Z0-9]", "", dataset_name.lower()) == "wikismall"
        ):  # control spelling of wiki small
            url = "https://github.com/data-iitd/tigger/raw/main/data/CAW_data/wiki_744_50.csv"
            # Load the CSV file into a DataFrame
            df = pd.read_csv(url, sep=",", header=None)
            df = df.iloc[:, 1:]

        else:
            raise ValueError(f"Invalid coresponding data to {dataset_name}.")
        # cleaned
        clean_df = df.drop(0)
        # Iterate over snapshots to construct the tgt graph dataset
        for snapshot in sorted(np.unique(clean_df.iloc[:, -1].to_numpy().astype(int))):
            data = (
                clean_df[clean_df.iloc[:, -1].to_numpy().astype(int) == snapshot]
                .iloc[:, :-1]
                .to_numpy()
                .astype(int)
            )

            edge_index = data[:, :2].T

            edge_weight = np.ones(edge_index[0].shape[0])
            if nb_nodes < edge_index.max():
                nb_nodes = edge_index.max() + 1
            if is_undirected:
                # double edges other way around
                edge_index = (
                    np.concatenate((edge_index, edge_index[[1, 0], :]), axis=1) - 1
                )
                edge_weight = np.concatenate((edge_weight, edge_weight))

            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)

        features = []
        targets = []
        for _ in range(len(edge_weights)):
            features.append(np.ones(nb_nodes).reshape([nb_nodes, 1]))
            targets.append(np.ones(nb_nodes))

        # Regroup into dynamic graph object
        dynamic_graph = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
        )

        return dynamic_graph


def from_tgt_to_networkx(tgt_dataset: DynamicGraphTemporalSignal) -> list:
    """
    Converts a tgt dataset into a list of networkX snapshots

    Args:
        tgt_dataset (Union[STATIC_GRAPH, DYNAMIC_GRAPH]): dataset to convert

    Returns:
        list[NETWORKX_GRAPH]: list of snapshots
    """
    max_timestamp = tgt_dataset.snapshot_count
    graph_list = [nx.Graph() for _ in range(0, max_timestamp)]

    for i, batch in enumerate(tgt_dataset):
        nb_nodes = batch.x.shape[0]
        for n in range(nb_nodes):
            graph_list[i].add_node(n, node_feat=batch.x[n, :])
        for e_i, edge in enumerate(batch.edge_index.T):
            n1, n2 = edge
            graph_list[i].add_edge(
                n1.item(), n2.item(), feat=batch.edge_attr[e_i].item()
            )

    return graph_list


def load_TwitterTennis_without_node_feat()-> DynamicGraphTemporalSignal:
    """_summary_

    Returns:
        DynamicGraphTemporalSignal: _description_
    """
    dynamic_signal = TwitterTennisDatasetLoader().get_dataset()
    feat = []
    for snap in dynamic_signal:
        feat.append(np.ones((snap.x.shape[0], 1)))

    return DynamicGraphTemporalSignal(
        dynamic_signal.edge_indices,
        dynamic_signal.edge_weights,
        feat,
        dynamic_signal.targets,
    )


def load_EnglandCovid_without_node_feat()-> DynamicGraphTemporalSignal:
    """_summary_

    Returns:
        DynamicGraphTemporalSignal: _description_
    """
    dynamic_signal = EnglandCovidDatasetLoader().get_dataset()
    feat = []
    for snap in dynamic_signal:
        feat.append(np.ones((snap.x.shape[0], 1)))
    return DynamicGraphTemporalSignal(
        dynamic_signal.edge_indices,
        dynamic_signal.edge_weights,
        feat,
        dynamic_signal.targets,
    )
