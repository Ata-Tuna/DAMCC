
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

import torch
import pytest
import tempfile
import numpy as np
from brainiac_temporal.metrics.nndr import (
    get_checkpoint ,
    get_snapshots_embeddings,
    get_snapshot_embeddings,
    transform_embeddings_matrix,
    load_lp_embedder,
)
from brainiac_temporal.models import LinkPredictor
from torch_geometric.data import Data
from typing import Iterable


@pytest.fixture
def fake_dataset() -> Iterable[Data]:
    """Make fake snapshots dataset

    Returns:
        Iterable[Data]: dataset of two snapshots
    """
    # Create a fake dataset using PyTorch Geometric Data
    num_nodes = 10
    x = torch.rand(num_nodes, 16)  # Features
    edge_index = torch.randint(0, num_nodes, (2, 20))  # Edge indices
    data = Data(x=x, edge_index=edge_index)
    return [data, data]  # Fake dataset with two snapshots


@pytest.fixture
def fake_link_predictor() -> LinkPredictor:
    """Make fake link predictor

    Returns:
        LinkPredictor: Embedder
    """
    # Create a fake LinkPredictor model
    return LinkPredictor(
        node_features=16,
        embedding_size=64,
        mlp_hidden_sizes=[32, 16],
        message_passing_class="GConvGRU",
        message_passing_kwargs={"K": 2},
    )

def test_get_checkpoint(fake_link_predictor: LinkPredictor) -> None:
    """Tests loading a checkpoint using _get_checkpoint

    Args:
        fake_link_predictor (LinkPredictor): Embedder
    """
    # Create a temporary directory for saving the checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = f"{temp_dir}/fake_embedder.pth"
        checkpoint = {
            "state_dict": fake_link_predictor.state_dict(),
            "epochs": 16,
            "embedding_size": 64,
            "mlp_hidden_sizes": [32, 16],
            "message_passing_class": "GConvGRU",
            "message_passing_kwargs": {"K": 2},
        }
        # Save the embedder to the temporary directory
        torch.save(checkpoint, checkpoint_path)

        # Load the checkpoint using _get_checkpoint
        loaded_checkpoint = get_checkpoint(checkpoint_path)

        # Compare values
        assert loaded_checkpoint["epochs"] == 16
        assert loaded_checkpoint["embedding_size"] == 64
        assert loaded_checkpoint["mlp_hidden_sizes"] == [32, 16]
        assert loaded_checkpoint["message_passing_class"] == "GConvGRU"
        assert loaded_checkpoint["message_passing_kwargs"] == {"K": 2}

def test_save_and_load_embedder(
    fake_link_predictor: LinkPredictor, fake_dataset: Iterable[Data]
) -> None:
    """Tests loading a checkpoint of a link predictor


    Args:
        fake_link_predictor (LinkPredictor): Embedder
        fake_dataset (Iterable[Data]): Snapshots
    """
    # Create a temporary directory for saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = f"{temp_dir}/fake_embedder.pth"
        checkpoint = {
            "state_dict": fake_link_predictor.state_dict(),
            "epochs": 16,
            "embedding_size": 64,
            "mlp_hidden_sizes": [32, 16],
            "message_passing_class": "GConvGRU",
            "message_passing_kwargs": {"K": 2},
        }
        # Save the embedder to the temporary directory
        torch.save(checkpoint, checkpoint_path)

        # Load the embedder back from the temporary directory
        loaded_embedder = load_lp_embedder(checkpoint_path, fake_dataset)

        # Perform your assertions here
        assert isinstance(loaded_embedder, LinkPredictor)


def test_get_snapshot_embeddings(
    fake_link_predictor: LinkPredictor, fake_dataset: Iterable[Data]
) -> None:
    """Tests  a single snapshot embeddings

    Args:
        fake_link_predictor (LinkPredictor): _description_
        fake_dataset (Iterable[Data]): _description_
    """
    snapshot = next(iter(fake_dataset))  # Use the first snapshot
    embeddings = get_snapshot_embeddings(fake_link_predictor, snapshot)
    assert isinstance(embeddings, torch.Tensor)


def test_get_snapshots_embeddings(
    fake_link_predictor: LinkPredictor, fake_dataset: Iterable[Data]
) -> None:
    """Tests getting all snapshots embeddings

    Args:
        fake_link_predictor (LinkPredictor): _description_
        fake_dataset (Iterable[Data]): _description_
    """
    embeddings = get_snapshots_embeddings(fake_link_predictor, fake_dataset)
    assert isinstance(embeddings, np.ndarray)


def test_transform_embeddings_matrix_dtw() -> None:
    """dtw transformation on embeddings matrix with dtw"""
    orig_embeddings = np.random.rand(5, 10, 16)
    gen_embeddings = np.random.rand(5, 10, 16)
    emb_matrix = transform_embeddings_matrix(orig_embeddings, gen_embeddings , nndr_calculation_method = "dtw")
    assert isinstance(emb_matrix, list)

def test_transform_embeddings_matrix_l2() -> None:
    """dtw transformation on embeddings matrix with l2"""
    orig_embeddings = np.random.rand(5, 10, 16)
    gen_embeddings = np.random.rand(5, 10, 16)
    emb_matrix = transform_embeddings_matrix(orig_embeddings, gen_embeddings , nndr_calculation_method = "l2")
    assert isinstance(emb_matrix, list)

def test_transform_embeddings_matrix_l1() -> None:
    """dtw transformation on embeddings matrix with l1"""
    orig_embeddings = np.random.rand(5, 10, 16)
    gen_embeddings = np.random.rand(5, 10, 16)
    emb_matrix = transform_embeddings_matrix(orig_embeddings, gen_embeddings , nndr_calculation_method = "l1")
    assert isinstance(emb_matrix, list)


def test_transform_embeddings_matrix_unknown_method()->None:
    """Test with unknown method
    """
    orig_embeddings = np.random.rand(5, 10, 16)
    gen_embeddings = np.random.rand(5, 10, 16)

    with pytest.raises(ValueError):
        transform_embeddings_matrix(orig_embeddings, gen_embeddings, nndr_calculation_method="unknown_method")
