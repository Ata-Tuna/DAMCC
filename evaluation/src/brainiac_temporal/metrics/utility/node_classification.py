
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

__all__ = ["node_classification_metric"]

import numpy as np
import pytorch_lightning as pl
from torch_geometric_temporal import DynamicGraphTemporalSignal  # type: ignore

from brainiac_temporal.data import LightningTemporalDataset
from brainiac_temporal.models.node_classifier import NodeClassifier


def node_classification_metric(
    original: DynamicGraphTemporalSignal,
    generated: DynamicGraphTemporalSignal,
    max_epochs: int = 1000,
    node_classifier_embedding_size: int = 32,
    node_classifier_mlp_hidden_sizes: list = [32, 32],
) -> float:
    """
    Computes the ROC-AUC score of a classification task on the node level, trained on the generated graph and tested on the original graph.

    Args:
        original (DynamicGraphTemporalSignal): The original graph.
        generated (DynamicGraphTemporalSignal): The generated graph.
        max_epochs (int, optional): The maximum number of epochs to train the model. Defaults to 1000.
        node_classifier_hidden_sizes (list, optional): The hidden sizes of the MLP. Defaults to [32, 32].
        node_classifier_K (int, optional): The Chebyshev filter size. Defaults to 2.

    Returns:
        float: The ROC-AUC score.
    """

    datamodule = LightningTemporalDataset(
        train_dataset=generated,
        val_dataset=original,
        test_dataset=original,
    )

    node_classifier = NodeClassifier(
        original.features[0].shape[1],
        len(np.unique(original.targets[0])),
        embedding_size=node_classifier_embedding_size,
        mlp_hidden_sizes=node_classifier_mlp_hidden_sizes,
        message_passing_class="GConvGRU",
        message_passing_kwargs={"K": 2},
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        auto_lr_find=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        replace_sampler_ddp=False,
    )

    trainer.fit(node_classifier, datamodule)

    test_results = trainer.test(node_classifier, datamodule)[0]

    return float(test_results["auc/test"])
