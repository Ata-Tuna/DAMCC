
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

__all__ = ["LightningTemporalDataset"]

from typing import Optional

import pytorch_lightning as pl
from torch_geometric_temporal import DynamicGraphTemporalSignal


class LightningTemporalDataset(pl.LightningDataModule):
    """LightningTemporalDataset"""

    def __init__(
        self,
        train_dataset: DynamicGraphTemporalSignal,
        val_dataset: Optional[DynamicGraphTemporalSignal] = None,
        test_dataset: Optional[DynamicGraphTemporalSignal] = None,
    ):
        """
        Wrapper to make temporal graphs compatible with lightning.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def prepare_data(self) -> None:
        """
        Prepare data for use in train, val, test.
        """
        super().prepare_data()

    def setup(self, stage: str) -> None:
        """
        Split into train, val, test.
        """
        super().setup(stage)

    def train_dataloader(self) -> DynamicGraphTemporalSignal:  # type: ignore
        """
        Return the train dataloader.
        """
        return self.train_dataset

    def val_dataloader(self) -> Optional[DynamicGraphTemporalSignal]:  # type: ignore
        """
        Return the validation dataloader.
        """
        return self.val_dataset

    def test_dataloader(self) -> Optional[DynamicGraphTemporalSignal]:  # type: ignore
        """
        Return the test dataloader.
        """
        return self.test_dataset
