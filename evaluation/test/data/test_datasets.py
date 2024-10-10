
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

from torch_geometric_temporal import DynamicGraphTemporalSignal
import torch_geometric_temporal as tgt
import networkx as nx
import torch
import pytest
from brainiac_temporal.data.datasets import (
    fetch_insecta_dataset,
    load_imdb_dynamic_tgt,
    load_tigger_datasets_tgt,
    from_tgt_to_networkx,
    load_EnglandCovid_without_node_feat,
    load_TwitterTennis_without_node_feat,
)


def test_insecta_loading() -> None:
    """Test loading of insecta dataset."""
    data = fetch_insecta_dataset()
    assert isinstance(data, DynamicGraphTemporalSignal)


def test_imdb_loading() -> None:
    """Test loading of imdb dataset."""
    data = load_imdb_dynamic_tgt()
    assert isinstance(data, DynamicGraphTemporalSignal)


def test_imdb_snapshot_count() -> None:
    """Test the number of snapshots in imdb dataset."""
    data = load_imdb_dynamic_tgt()
    assert data.snapshot_count == 28


def test_wiki_loading() -> None:
    """Test loading of wiki dataset."""
    data = load_tigger_datasets_tgt(dataset_name="WIKI_small")
    assert isinstance(data, DynamicGraphTemporalSignal)


def test_wiki_spelling() -> None:
    """Test loading of wiki dataset."""
    data = load_tigger_datasets_tgt(dataset_name="WIKI_small")
    assert isinstance(data, DynamicGraphTemporalSignal)


def test_bitcoin_loading() -> None:
    """Test loading of reddit dataset."""
    data = load_tigger_datasets_tgt(dataset_name="Bitcoin")
    assert isinstance(data, DynamicGraphTemporalSignal)


def test_wiki_snapshot_count() -> None:
    """Test loading of wiki dataset."""
    data = load_tigger_datasets_tgt(dataset_name="WIKI_small")
    assert data.snapshot_count == 50


def test_bitcoin_snapshot_count() -> None:
    """Test loading of reddit dataset."""
    data = load_tigger_datasets_tgt(dataset_name="Bitcoin")
    assert data.snapshot_count == 190


def test_tgt_to_networkx_converter() -> None:
    """ "Test tgt to netwrokx converter"""
    dataset_tgt = tgt.EnglandCovidDatasetLoader().get_dataset()

    networkX_graph_list = from_tgt_to_networkx(dataset_tgt)

    for i, batch in enumerate(dataset_tgt):

        # the same number of nodes
        assert batch.x.shape[0] == networkX_graph_list[i].number_of_nodes()

        # the same node features
        assert torch.all(
            batch.x[0, :].eq(
                nx.get_node_attributes(networkX_graph_list[i], "node_feat")[0]
            )
        )


def test_twitter_tennis_dataset_shape() -> None:
    """_summary_"""
    twitter_tennis_dataset = load_TwitterTennis_without_node_feat()
    reference_dataset = tgt.TwitterTennisDatasetLoader().get_dataset()
    # check carateristics that will be feed to the embedder
    assert twitter_tennis_dataset.snapshot_count == reference_dataset.snapshot_count
    for snap in twitter_tennis_dataset:
        assert snap.x.shape[1] == 1  # Check if the node feature dimension is 1


def test_england_covid_dataset_shape() -> None:
    """_summary_"""
    england_covid_dataset = load_EnglandCovid_without_node_feat()
    reference_dataset = tgt.EnglandCovidDatasetLoader().get_dataset()
    # check carateristics that will be feed to the embedder
    assert england_covid_dataset.snapshot_count == reference_dataset.snapshot_count
    for snap in england_covid_dataset:
        assert snap.x.shape[1] == 1  # Check if the node feature dimension is 1


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
