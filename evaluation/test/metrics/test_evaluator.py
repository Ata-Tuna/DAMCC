
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
import torch_geometric_temporal as tgt
import tempfile
import torch
from brainiac_temporal.metrics.evaluator import MetricEvaluator
from brainiac_temporal.models import LinkPredictor
import pytest

# Test nndr

@pytest.fixture
def fake_link_predictor() -> LinkPredictor:
    """Make fake link predictor

    Returns:
        LinkPredictor: Embedder
    """
    # Create a fake LinkPredictor model
    return LinkPredictor(
        node_features= (tgt.EnglandCovidDatasetLoader().get_dataset())[0].x.size(1),
        embedding_size=16,
        mlp_hidden_sizes=[32, 16],
        message_passing_class="GConvGRU",
        message_passing_kwargs={"K": 2},
    )


def test_nndr_mteric_in_evaluator(
    fake_link_predictor: LinkPredictor
) -> None:
    """Tests nndr metric in evaluator"""
    # Create a temporary directory for saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = f"{temp_dir}/fake_embedder.pth"
        checkpoint = {
            "state_dict": fake_link_predictor.state_dict(),
            "epochs": 16,
            "embedding_size": 16,
            "mlp_hidden_sizes": [32, 16],
            "message_passing_class": "GConvGRU",
            "message_passing_kwargs": {"K": 2},
        }
        # Save the embedder to the temporary directory
        torch.save(checkpoint, checkpoint_path)
        #try with dtw method
        metric_evaluator = MetricEvaluator(
            utility_metrics=None, get_privacy_metric=True, embedder_path=checkpoint_path, nndr_calculation_method = "dtw",
        )

        original = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]
        generated = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]

        results = metric_evaluator(original, generated)
        assert isinstance(results["NNDR_mean"], float)
        assert isinstance(results["NNDR_std"], float)
        #try with l1 method
        metric_evaluator = MetricEvaluator(
            utility_metrics=None, get_privacy_metric=True, embedder_path=checkpoint_path, nndr_calculation_method = "l1",
        )

        original = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]
        generated = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]

        results = metric_evaluator(original, generated)
        assert isinstance(results["NNDR_mean"], float)
        assert isinstance(results["NNDR_std"], float)
        #try with l2 method
        metric_evaluator = MetricEvaluator(
                    utility_metrics=None, get_privacy_metric=True, embedder_path=checkpoint_path, nndr_calculation_method = "l2",
                )

        original = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]
        generated = tgt.EnglandCovidDatasetLoader().get_dataset()[:2]

        results = metric_evaluator(original, generated)
        assert isinstance(results["NNDR_mean"], float)
        assert isinstance(results["NNDR_std"], float)



def test_evaluator() -> None:
    """Tests the Evaluator class."""
    dataset_tgt = tgt.TwitterTennisDatasetLoader().get_dataset()

    for i, batch in enumerate(dataset_tgt.targets):
        dataset_tgt.targets[i] = np.ones(batch.shape[0])
        dataset_tgt.targets[i][0] = 0

    generated = dataset_tgt
    original = dataset_tgt

    metric_evaluator = MetricEvaluator(utility_metrics_kwargs={"max_epochs": 2})
    metrics = metric_evaluator(original, generated)

    for name, value in metrics.items():
        print(name)
        if name == "node_classification":
            assert value > 0.5
        elif name == "link_forecasting":
            assert value > 0.5
        elif name == "temporal_correlation":
            assert value == 0.0
        else:
            assert value == 0.0 or np.isnan(value)


def test_evaluator_different_sequences() -> None:
    """Tests the Evaluator class with very different sequences."""
    dataset_original = tgt.EnglandCovidDatasetLoader().get_dataset()
    dataset_generated = tgt.TwitterTennisDatasetLoader().get_dataset()

    shape = -1

    for i, batch in enumerate(dataset_original.targets):
        if shape == -1:
            shape = dataset_original.features[i].shape[1]
        dataset_original.targets[i] = np.ones(batch.shape[0])
        dataset_original.targets[i][0] = 0

    for i, batch in enumerate(dataset_generated.targets):
        dataset_generated.targets[i] = np.ones(batch.shape[0])
        dataset_generated.targets[i][0] = 0

        dataset_generated.features[i] = np.ones((batch.shape[0], shape))

    generated = dataset_generated
    original = dataset_original

    metric_evaluator = MetricEvaluator(
        utility_metrics_kwargs={"max_epochs": 2},
    )
    metrics = metric_evaluator(original, generated)

    for name, value in metrics.items():
        if name == "node_classification":
            assert value < 0.9
        elif name == "link_forecasting":
            assert value < 0.9
        elif name == "temporal_correlation":
            assert value > 0.1
        else:
            assert value > 0.1 or np.isnan(value)


def test_evaluator_without_targets() -> None:
    """Tests the Evaluator class with sequences without node features."""
    dataset_original = tgt.TwitterTennisDatasetLoader().get_dataset()
    dataset_generated = tgt.TwitterTennisDatasetLoader().get_dataset()

    for i in range(len(dataset_original.targets)):
        dataset_original.targets[i] = None

    for i in range(len(dataset_generated.targets)):
        dataset_generated.targets[i] = None

    generated = dataset_generated
    original = dataset_original

    metric_evaluator = MetricEvaluator(
        utility_metrics_kwargs={"max_epochs": 2},
    )
    metrics = metric_evaluator(original, generated)

    assert "node_classification" not in metrics

    for name, value in metrics.items():
        if name == "link_forecasting":
            assert value > 0.7
        elif name == "temporal_correlation":
            assert value == 0.0
        else:
            assert value == 0.0 or np.isnan(value)


def test_evaluator_without_node_features() -> None:
    """Tests the Evaluator class with sequences without node features."""
    dataset_original = tgt.TwitterTennisDatasetLoader().get_dataset()
    dataset_generated = tgt.TwitterTennisDatasetLoader().get_dataset()

    for i, batch in enumerate(dataset_original.targets):
        dataset_original.targets[i] = np.ones(batch.shape[0])
        dataset_original.targets[i][0] = 0

        dataset_original.features[i] = np.ones((batch.shape[0], 1))

    for i, batch in enumerate(dataset_generated.targets):
        dataset_generated.targets[i] = np.ones(batch.shape[0])
        dataset_generated.targets[i][0] = 0

        dataset_generated.features[i] = np.ones((batch.shape[0], 1))

    generated = dataset_generated
    original = dataset_original

    metric_evaluator = MetricEvaluator(
        utility_metrics_kwargs={"max_epochs": 2},
    )
    metrics = metric_evaluator(original, generated)

    for name, value in metrics.items():
        if name == "link_forecasting":
            assert value > 0.3
        elif name == "node_classification":
            assert value > 0.3
        else:
            assert value == 0.0 or np.isnan(value)
