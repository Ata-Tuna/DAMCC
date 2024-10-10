
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

__all__ = ["LinkPredictor"]

from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric_temporal.nn import GConvGRU
from torchmetrics import AUROC
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from brainiac_temporal.models.lightning import TemporalLightningModule


class LinkPredictor(TemporalLightningModule):
    """Link Predictor Model."""

    def __init__(
        self,
        node_features: int,
        embedding_size: int,
        mlp_hidden_sizes: List[int],
        message_passing_class: str = "GConvGRU",
        message_passing_kwargs: Dict[str, Any] = {},
        lr: float = 1e-3,
    ) -> None:
        """
        Args:
            node_features (int): _description_
            embedding_size (int): _description_
            mlp_hidden_sizes (List[int]): _description_
            message_passing_class (str, optional): _description_. Defaults to "GConvGRU".
            message_passing_kwargs (Dict[str, Any], optional): _description_. Defaults to {}.
            lr (float, optional): _description_. Defaults to 1e-3.
        """
        super(LinkPredictor, self).__init__()
        in_dim = node_features

        if message_passing_class == "GConvGRU":
            message_passing_initializer = GConvGRU
        else:
            raise ValueError("Invalid message passing class")

        self.recurrent = message_passing_initializer(
            in_channels=in_dim, out_channels=embedding_size, **message_passing_kwargs
        )

        self.mlp = torch.nn.ModuleList()

        in_dim = embedding_size * 2
        for layer_size in mlp_hidden_sizes:
            self.mlp.append(torch.nn.Linear(in_dim, layer_size))
            in_dim = layer_size
        self.mlp.append(torch.nn.Linear(in_dim, 1))

        self.auc = AUROC(task="binary")

        self.lr = lr
        self.h: Optional[torch.Tensor] = None

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        s: torch.Tensor = self.recurrent(x, edge_index, H=h)
        s = torch.relu(s)
        return s

    def predict(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        use_sigmoid: bool = False,
    ) -> torch.Tensor:
        """Predicts the probability of a link."""
        h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        for i, layer in enumerate(self.mlp):
            h = layer(h)
            if i < len(self.mlp) - 1:
                h = torch.relu(h)

        if use_sigmoid:
            h = torch.sigmoid(h)

        return h

    def loss(
        self,
        batch: Data,
        stage: str,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the loss."""
        neg_edge_index = negative_sampling(
            edge_index=batch.edge_index, num_nodes=batch.num_nodes
        )
        h = self.forward(batch.x, batch.edge_index, h)
        pos_out = self.predict(h, batch.edge_index)
        neg_out = self.predict(h, neg_edge_index)
        y = torch.cat(
            [torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])], dim=0
        )
        y_hat = torch.cat([pos_out, neg_out], dim=0).reshape(-1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

        self.log(f"loss/{stage}", loss, prog_bar=True)

        if stage == "val" or stage == "test":
            self.log(f"auc/{stage}", self.auc(y_hat, y.long()), prog_bar=True)

        self.h = h.clone().detach()
        return loss

    def training_step(self, batch: Data, *args) -> torch.Tensor:  # type: ignore
        """Computes the loss on the training set."""
        loss = self.loss(batch, "train", self.h)
        return loss

    def validation_step(self, batch: Data, *args) -> torch.Tensor:  # type: ignore
        """Computes the loss on the validation set."""
        loss = self.loss(batch, "val", self.h)
        return loss

    def test_step(self, batch: Data, *args) -> torch.Tensor:  # type: ignore
        """Computes the loss on the test set."""
        loss = self.loss(batch, "test", self.h)
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
