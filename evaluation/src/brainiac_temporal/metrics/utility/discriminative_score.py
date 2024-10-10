
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

__all__ = ["discriminative_score_metric"]

from typing import Sequence

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from tqdm import tqdm

from brainiac_temporal.data import GraphTimeseriesDataset
from brainiac_temporal.models.recurrent_gcn import RecurrentGCN
from brainiac_temporal.types import GRAPH_SERIES


def discriminative_score_metric(
    original: Sequence[GRAPH_SERIES],
    generated: Sequence[GRAPH_SERIES],
    epochs: int = 10,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    train_ratio: float = 0.8,
    # **kwargs: Any,
) -> float:
    """
    Computes the classification error of a trained model to classify real and synthetic (generated) temporal graphs.
    Args:
        original (Sequence[GRAPH_SERIES]):
            List of original graphs. Each graph is represented by a tuple `(x, edge_index, edge_weight)`.
        generated (Sequence[GRAPH_SERIES]):
            List of generated graphs. Each graph is represented by a tuple `(x, edge_index, edge_weight)`.
        epochs (int): the number of epochs.
        learning_rate (float): training learning rate.
        weight_decay (float): training weight decay.
        train_ratio (float): the percentage of data used for training.
    Returns:
        float: The percentage of misclassified samples.
    """

    # init params
    node_count = original[0][0][0].shape[0]
    num_features = original[0][0][0].shape[1]

    # Split data into train and test
    dataset = GraphTimeseriesDataset(original, generated)
    tot = len(dataset)
    n_train = int(train_ratio * tot)
    n_val = tot - n_train
    train_data, test_data = random_split(dataset, [n_train, n_val])  # type: ignore

    # Graph time-series classifier
    model = RecurrentGCN(
        num_features=num_features,
        num_classes=2,
        node_count=node_count,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Training
    cfg = model.configure_optimizers()
    optimizer: torch.optim.Optimizer = cfg["optimizer"]
    scheduler: ReduceLROnPlateau = cfg["lr_scheduler"]["scheduler"]
    for epoch in range(epochs):
        total_loss = 0.0
        batch_idx = 0
        for batch_idx, batch in tqdm(enumerate(train_data), desc=f"Training (epoch={epoch})"):  # type: ignore
            loss = model.training_step(batch, batch_idx)
            loss.backward()  # type:ignore
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss  # type: ignore
        scheduler.step(total_loss / batch_idx)

    # Testing
    model.accuracy.reset()
    for batch_idx, batch in tqdm(enumerate(test_data), desc="Testing"):  # type: ignore
        model.test_step(batch, batch_idx)
    acc = model.accuracy.compute()

    # Return error
    return 1 - float(acc)
