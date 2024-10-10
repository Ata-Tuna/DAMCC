
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

import abc
import typing as ty
from typing import Optional

import pytorch_lightning as pl
import torch


class TemporalLightningModule(pl.LightningModule, abc.ABC):
    """Base class for temporal GNN models."""

    def __init__(
        self,
        lr: float = 1e-3,
    ) -> None:
        """Temporal Lightning Module interface."""
        super().__init__()
        self.lr: float = lr
        self.h: Optional[torch.Tensor] = None

    def on_train_epoch_start(self) -> None:
        """Resets the hidden state after an epoch."""
        self.h = None

    def on_validation_epoch_start(self) -> None:
        """Resets the hidden state after an epoch."""
        self.h = None

    def on_test_epoch_start(self) -> None:
        """Resets the hidden state after an epoch."""
        self.h = None

    def on_predict_epoch_start(self) -> None:
        """Resets the hidden state after an epoch."""
        self.h = None
