
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

import numpy as np
import pytorch_lightning as pl
import torch_geometric_temporal as tgt  # type: ignore

from brainiac_temporal.data import LightningTemporalDataset
from brainiac_temporal.models import NodeClassifier


def test_node_classifier() -> None:
    """Tests the NodeClassifier model."""

    dataset_tgt = tgt.TwitterTennisDatasetLoader().get_dataset()

    for i, batch in enumerate(dataset_tgt.targets):
        dataset_tgt.targets[i] = np.ones(batch.shape[0])
        dataset_tgt.targets[i][0] = 0

    datamodule = LightningTemporalDataset(
        train_dataset=dataset_tgt,
        val_dataset=dataset_tgt,
        test_dataset=dataset_tgt,
    )

    model = NodeClassifier(
        dataset_tgt.features[0].shape[1],
        len(np.unique(dataset_tgt.targets[0])),
        32,
        [32, 32],
        message_passing_class="GConvGRU",
        message_passing_kwargs={"K": 2},
    )

    trainer = pl.Trainer(
        max_epochs=2,
        auto_lr_find=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
