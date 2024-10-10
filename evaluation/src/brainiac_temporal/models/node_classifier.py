
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

__all__ = ["NodeClassifier"]

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch  # type: ignore
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torchmetrics import AUROC

from brainiac_temporal.models.lightning import TemporalLightningModule


class NodeClassifier(TemporalLightningModule):
    """Node classifier."""

    def __init__(
        self,
        node_features: int,
        num_classes: int,
        embedding_size: int,
        mlp_hidden_sizes: List[int],
        message_passing_class: str = "GConvGRU",
        message_passing_kwargs: Dict[str, Any] = {},
        lr: float = 1e-3,
    ):
        """Recurrent GCN model.

        Args:
            node_features (int): dimension of the node features
            num_classes (int): number of classes
            K (int): Chebychev filter size
            normalization (str, optional): normalization method. Defaults to "sym".
            bias (bool, optional): whether to learn a bias or not. Defaults to True.
        """
        super(NodeClassifier, self).__init__()
        in_dim = node_features

        if message_passing_class == "GConvGRU":
            message_passing_initializer = GConvGRU
        else:
            raise ValueError("Invalid message passing class")

        self.recurrent = message_passing_initializer(
            in_channels=in_dim, out_channels=embedding_size, **message_passing_kwargs
        )

        self.mlp = torch.nn.ModuleList()

        in_dim = embedding_size
        for layer_size in mlp_hidden_sizes:
            self.mlp.append(torch.nn.Linear(in_dim, layer_size))
            in_dim = layer_size
        self.mlp.append(torch.nn.Linear(in_dim, num_classes))

        self.auc = AUROC(num_classes=num_classes, task="multiclass")

        self.lr = lr
        self.h: Optional[torch.Tensor] = None

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): The node features of the graph
            edge_index (torch.Tensor): The edge indices of the graph
            edge_weight (torch.Tensor): The edge weights of the graph

        Returns:
            torch.Tensor : The class logits for the nodes.
        """
        s = x
        s = self.recurrent(s, edge_index, edge_weight, h)
        s = F.relu(s)
        self.h = s.clone().detach()
        for i, layer in enumerate(self.mlp):
            s = layer(s)
            if i < len(self.mlp) - 1:
                s = torch.relu(s)
        return s

    def base_step(
        self,
        batch: Batch,
        stage: str,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Common step."""
        x, edge_index, edge_weight, y = (
            batch.x,
            batch.edge_index,
            batch.edge_weight,
            batch.y,
        )
        y_hat = self(x, edge_index, edge_weight, h)
        loss = F.cross_entropy(y_hat, y.long())
        self.log(f"CE/{stage}", loss)

        if stage == "val" or stage == "test":
            self.log(f"auc/{stage}", self.auc(y_hat, y.long()), prog_bar=True)
        return loss

    def training_step(self, batch: Batch, *args) -> torch.Tensor:  # type: ignore
        """Training step."""
        loss = self.base_step(batch, "train", self.h)
        return loss

    def validation_step(self, batch: Batch, *args) -> torch.Tensor:  # type: ignore
        """Validation step."""
        loss = self.base_step(batch, "val", self.h)
        return loss

    def test_step(self, batch: Batch, *args) -> torch.Tensor:  # type: ignore
        """Test step."""
        loss = self.base_step(batch, "test", self.h)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
                "strict": False,
            },
        }

    def configure_callbacks(self) -> List[pl.Callback]:  # type: ignore
        """Configure checkpoint."""
        callbacks = []
        ckpt_cb_val: pl.Callback = cb.ModelCheckpoint(
            monitor="auc/val",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks.append(ckpt_cb_val)
        early: pl.Callback = cb.EarlyStopping(
            monitor="loss/train",
            patience=5,
            mode="min",
            check_on_train_epoch_end=True,
            strict=False,
        )
        callbacks.append(early)
        return callbacks
