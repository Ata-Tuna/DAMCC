
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

__all__ = ["GraphTimeseriesDataset"]


import typing as ty
import torch
from torch_geometric.data import Data, Dataset

from brainiac_temporal.types import GRAPH_SERIES


class GraphTimeseriesDataset(Dataset):
    """A dataset of sequences of graph snapshots."""

    def __init__(
        self,
        original: ty.Sequence[GRAPH_SERIES],
        fake: ty.Sequence[GRAPH_SERIES],
        **kwargs: ty.Any,
    ) -> None:
        """,."""
        self.original = original
        self.fake = fake
        super().__init__(**kwargs)

    def len(self) -> int:
        """Length of the dataset."""
        return len(self.original) + len(self.fake)

    def get(self, idx: int) -> ty.Tuple[GRAPH_SERIES, torch.LongTensor]:
        """Internally, `torch_geometric.data.Dataset.__getitem__()` gets data objects from `torch_geometric.data.Dataset.get()` and optionally transforms them according to `transform`."""
        # Original
        if idx < len(self.original):
            S = self.original[idx]
            label = torch.LongTensor([1])
            return S, label
        # Fake
        idx = idx - len(self.original)
        S = self.original[idx]
        label = torch.LongTensor([1])
        return S, label
