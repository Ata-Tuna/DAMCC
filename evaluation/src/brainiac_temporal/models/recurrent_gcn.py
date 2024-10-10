
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

__all__ = ["RecurrentGCN"]

from loguru import logger
import typing as ty
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Accuracy, Metric
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from brainiac_temporal.types import GRAPH, GRAPH_SERIES


class RecurrentGCN(pl.LightningModule):
    """Recurrent GCN model"""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        node_count: int,
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-4,
        patience: int = 2,
    ) -> None:
        """Recurrent GCN model.
        Args:
            num_features (int): dimension of the node features
            num_classes (int): number of classes
            node_count (int): Chebychev filter size
        """
        super().__init__()
        self.save_hyperparameters()
        # NN
        self.recurrent_1 = GConvLSTM(num_features, 32, 5)
        self.recurrent_2 = GConvLSTM(32, 16, 5)
        self.linear = torch.nn.Linear(16, 4)
        self.linear2 = torch.nn.Linear(4 * node_count, num_classes)
        # Params
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        # Metrics
        task = "multiclass" if num_classes > 2 else "binary"
        self.accuracy: Metric = Accuracy(task, num_classes=num_classes)  # type: ignore

    def forward(  # type: ignore
        self,
        graphs: GRAPH_SERIES,
    ) -> torch.Tensor:
        """Forward pass
        Args:
            graphs (list): graph snapshots

        Returns:
            torch.Tensor : The class logits for the nodes.
        """
        # Process the sequence of graphs with our 2 GConvLSTM layers
        # Initialize hidden and cell states to None so they are properly
        # initialized automatically in the GConvLSTM layers.
        h1, c1, h2, c2 = None, None, None, None
        for x, edge_index, edge_weight in graphs:
            h1, c1 = self.recurrent_1(x, edge_index, edge_weight, H=h1, C=c1)
            # Feed hidden state output of first layer to the 2nd layer
            h2, c2 = self.recurrent_2(h1, edge_index, edge_weight, H=h2, C=c2)
        # Use the final hidden state output of 2nd recurrent layer for input to classifier
        assert isinstance(h2, torch.Tensor)
        h: Tensor = F.relu(h2)
        h = F.dropout(h, training=self.training)
        h = self.linear(h)
        a = h.reshape(1, -1)
        result = self.linear2(a)
        return F.log_softmax(result, dim=1)

    def predict(self, graphs: GRAPH_SERIES) -> Tensor:
        """Predict function
        Args:
            graphs: (list): graph snapshots

        Returns:
            int : The class logits for the nodes.
        """
        return torch.argmax(self(graphs))

    def configure_optimizers(self) -> ty.Dict[str, ty.Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
                "strict": False,
            },
        }

    def configure_callbacks(self) -> pl.Callback:
        """Add Callbacks."""
        return EarlyStopping(
            monitor="loss/train",
            patience=10,
            strict=False,
            check_on_train_epoch_end=False,
            mode="min",
            verbose=False,
        )

    def training_step(  # type: ignore
        self,
        batch: ty.Tuple[GRAPH_SERIES, torch.LongTensor],
        batch_idx: int,
    ) -> Tensor:
        loss, _ = self.__step__(batch, batch_idx, "train")
        return loss

    def validation_step(  # type: ignore
        self,
        batch: ty.Tuple[GRAPH_SERIES, torch.LongTensor],
        batch_idx: int,
    ) -> Tensor:
        loss, _ = self.__step__(batch, batch_idx, "val")
        return loss

    def test_step(  # type: ignore
        self,
        batch: ty.Tuple[GRAPH_SERIES, torch.LongTensor],
        batch_idx: int,
    ) -> Tensor:
        loss, _ = self.__step__(batch, batch_idx, "test")
        return loss

    def __step__(
        self,
        batch: ty.Tuple[GRAPH_SERIES, torch.LongTensor],
        batch_idx: int,
        stage: str = None,
    ) -> ty.Tuple[Tensor, Tensor]:
        """Common step."""
        _explain_batch(batch, batch_idx)
        # Process batch
        tempralGRAPH, target = batch
        # Apply the model to the graph sequence
        scores: Tensor = self(tempralGRAPH)
        # Loss
        assert isinstance(target, (Tensor, torch.LongTensor))
        loss: Tensor = F.cross_entropy(scores, target)
        # Predict
        preds = scores.argmax()
        # Update metrics and log
        with torch.no_grad():
            self.accuracy.update(preds.detach().clone().view(-1), target.view(-1))  # type: ignore
            try:
                self.log(f"loss/{stage}", loss, prog_bar=True)
                self.log(f"acc/{stage}", self.accuracy, prog_bar=True)
            except Exception:
                pass
        return loss, preds


def _explain_batch(batch: ty.Any, batch_idx: int) -> None:
    """Explain batch."""
    if batch_idx > 0:
        return
    logger.trace(f"({batch_idx}): {type(batch)}")
    if isinstance(batch, (tuple, list)):
        for x in batch:
            _explain_batch(x, batch_idx)
