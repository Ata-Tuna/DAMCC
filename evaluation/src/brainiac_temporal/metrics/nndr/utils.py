
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
    "get_checkpoint",
    "load_lp_embedder",
    "get_snapshot_embeddings",
    "get_snapshots_embeddings",
    "transform_embeddings_matrix",
]

from typing import Any, Dict, Iterable, List
from torch_geometric.data import Data
import numpy as np
import torch
from brainiac_temporal.models import LinkPredictor
from dtaidistance.dtw_ndim import distance_fast

def get_checkpoint(path: str)-> Any:
    """Load checkpoint of embedder

    Args:
        path (str): embedder checkpoint path

    Returns:
        Any: dictionnary of the checkpoint
    """
    return torch.load(path) # type: ignore


def load_lp_embedder(path: str,dataset: Iterable[Data]) -> LinkPredictor:
    """Loads link predictor embedder from checkpoint

    Args:
        path (str): Path to embedder checkpoint
        dataset ( Iterable[Data]): dataset to extract its node features size

    Returns:
        LinkPredictor: Link Predictor Model
    """
    checkpoint = get_checkpoint(path)
    g= next(iter(dataset))
    node_features = g.x.shape[1]
    embedder = LinkPredictor(
        node_features=node_features,
        embedding_size=checkpoint["embedding_size"],
        mlp_hidden_sizes=checkpoint["mlp_hidden_sizes"],
        message_passing_class="GConvGRU",
        message_passing_kwargs=checkpoint["message_passing_kwargs"],
    )
    # Transfer weights from the checkpoint to the model
    embedder.load_state_dict(checkpoint["state_dict"])
    embedder.eval()
    return embedder


def get_snapshot_embeddings(
    model: LinkPredictor, snapshot: Data, prev_embedding: torch.Tensor = None
) -> Any:
    """Returns the embeddings for a snapshot

    Args:
        model (LinkPredictor): Embedder
        snapshot (Data): snapshot
        prev_embedding (torch.Tensor, optional): The previous embeddings to capture temporality. Defaults to None.

    Returns:
        Any: The embeddings of the next snapshot
    """
    with torch.no_grad():
        x, edge_index = snapshot.x, snapshot.edge_index
        # Apply the model to get embeddings
        embeddings = model(x, edge_index, prev_embedding)
    return embeddings


def get_snapshots_embeddings(
    embedder: LinkPredictor, snapshots_iterable: Iterable[Data]
) -> np.ndarray:
    """Get all snapshpots embeddings

    Args:
        embedder (LinkPredictor): Embedder
        list(snapshots_iterable) (Iterable[Data]): Full dataset snapshots

    Returns:
        np.ndarray: Snapshots embeddings
    """

    snapshots_list = list(snapshots_iterable)
    # Iterate through the dataset and compute embeddings
    embeddings = get_snapshot_embeddings(embedder, snapshots_list[0])
    embeddings_list = [embeddings.cpu().numpy()]
    for data in snapshots_list[1:]:
        if data.edge_index.size(0) != 0:
            embeddings = get_snapshot_embeddings(
                embedder, data, torch.tensor(embeddings)
            )
        embeddings_list.append(embeddings.cpu().numpy())

    return np.stack(embeddings_list)


def transform_embeddings_matrix(
    orig_embeddings: np.ndarray, gen_embeddings: np.ndarray, nndr_calculation_method:str ="dtw"
) -> List[list]:
    """Applies dtw on the original embeddings the generated ones over snapshots

    Args:
        orig_embeddings (np.ndarray): Original snapshots embeddings
        gen_embeddings (np.ndarray): Generated snapshots embeddings
        calculation_method (str): calculation methode of the nndr

    Returns:
        List[list]: Embeddings matrix
    """
    # distance_fast requires conversion to double
    gen_embeddings = np.array(gen_embeddings, dtype=np.float64)
    orig_embeddings = np.array(orig_embeddings, dtype=np.float64)
    # get matrix of embeddings
    if nndr_calculation_method == "dtw":
        emb_matrix = [
            [
                distance_fast(gen_embeddings[:, i, :], orig_embeddings[:, j, :])
                for j in range(orig_embeddings.shape[1])
            ]
            for i in range(gen_embeddings.shape[1])
        ]
    elif nndr_calculation_method in ["l1", "l2", "l_1", "l_2"]:
        last_gen_emb = gen_embeddings[-1]
        last_orig_emb = orig_embeddings[-1] #shape(num_nodes, emb_size)
        if nndr_calculation_method in ["l2", "l_2"]:
            # for each node calculate the distance
            emb_matrix = [np.linalg.norm(last_gen_emb[:, np.newaxis, :] - last_orig_emb, axis=-1)]
        else:
            emb_matrix = [np.sum(np.abs(last_gen_emb[:, np.newaxis, :] - last_orig_emb), axis=-1)]
    else:
        raise ValueError (f"Unknown method {nndr_calculation_method}. Please select valid nndr_calculation_method (dtw, l1 or l2).")
    return emb_matrix
