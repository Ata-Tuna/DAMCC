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

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from torch_geometric_temporal import TwitterTennisDatasetLoader
from torch_geometric_temporal.dataset.encovid import EnglandCovidDatasetLoader

from brainiac_temporal.data.datasets import (
    fetch_insecta_dataset,
    load_imdb_dynamic_tgt,
    load_tigger_datasets_tgt,
)
from brainiac_temporal.metrics import MetricEvaluator

sys.path.append("sota_paper_implementation")
from convertors.d2g22tgt import convert_d2g2_sample_to_tgt
from convertors.damnets2tgt import convert_damnets_sample_to_tgt
from convertors.dymond2tgt import convert_dymond_sample_to_tgt
from convertors.tigger2tgt import convert_tigger_sample_to_tgt

# The boolean tells if this is an update model
update_model = {
    "tigger": False,
    "random": True,
    "age": True,
    "damnets": True,
    "damcc": True,
    "dymond": False,
    "d2g2": False,
}
available_models = list(update_model.keys())
available_datasets = [
    "insecta",
    "wikismall",
    "twittertennis",
    "imdb",
    "bitcoin",
    "reddit",
    "encovid",
    "3comm",
    "ba"
]

def check_files_are_valid(files: List[Path]) -> None:
    """
    Checks that all the files in the list respect the naming convention and that for update models
    a test sequence is given with the generated sequence.
    """
    for file in files:
        extensionless_name = file.stem
        if extensionless_name in ["metrics", "not-run"]:
            continue

        # Split by "-" and validate
        name_parts = extensionless_name.split("-")
        if len(name_parts) != 2:
            raise ValueError(
                f"Incorrect file name format '{extensionless_name}'. Must be '[model]-[dataset].[extension]'."
            )

        model, dataset = name_parts
        if model not in available_models:
            raise ValueError(f"Model '{model}' not in available models. Available models: {available_models}")
        if dataset not in available_datasets:
            raise ValueError(f"Dataset '{dataset}' not in available datasets. Available datasets: {available_datasets}")

def load_reference_dataset(dataset_name: str, test_file_path: Optional[Path] = None) -> Union[Path, any]:
    """
    Loads the reference dataset based on the dataset name and test file path.
    """
    if test_file_path:
        return test_file_path  # Directly use the provided test file path
    
    # Dataset loading logic based on name
    if dataset_name == "insecta":
        return fetch_insecta_dataset(colony=6)
    elif dataset_name == "imdb":
        return load_imdb_dynamic_tgt()
    elif dataset_name in ["wikismall", "bitcoin", "reddit"]:
        return load_tigger_datasets_tgt(dataset_name)
    elif dataset_name == "encovid":
        return EnglandCovidDatasetLoader().get_dataset()
    elif dataset_name == "twittertennis":
        return TwitterTennisDatasetLoader().get_dataset()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

def main(files: Optional[List[Path]] = None, test_file_path: Optional[Path] = None) -> None:
    directory = Path("generated_graphs")

    # If no specific files are passed, use all files from the directory
    if files is None:
        files = [x for x in directory.iterdir() if x.is_file()]

    # Validate files
    check_files_are_valid(files)

    for file in files:
        extensionless_name = file.stem
        name_, note = extensionless_name, ""
        if "_" in extensionless_name:
            name_, note = extensionless_name.split("_")
        parts = name_.split("-")
        
        if extensionless_name in ["metrics", "not-run"]:
            continue

        model_name, dataset_name = parts

        # Select the converter based on the model
        converters = {
            "damnets": convert_damnets_sample_to_tgt,
            "age": convert_damnets_sample_to_tgt,
            "random": convert_damnets_sample_to_tgt,
            "damcc": convert_damnets_sample_to_tgt,
            "dymond": convert_dymond_sample_to_tgt,
            "d2g2": convert_d2g2_sample_to_tgt,
            "tigger": convert_tigger_sample_to_tgt,
        }

        converter = converters.get(model_name)
        if converter is None:
            raise ValueError(f"Unknown model '{model_name}'")

        # Load reference dataset
        reference_dataset = load_reference_dataset(dataset_name, test_file_path)
        generated_dataset = converter(file)

        print(f"Generated dataset for {model_name}-{dataset_name}:", generated_dataset)
        reference_dataset = converter(reference_dataset)
        print(f"Reference dataset for {model_name}-{dataset_name}:", reference_dataset)

        # Evaluate the metrics
        evaluator = MetricEvaluator(
            statistics="all",
            temporal_aggregation="dtw",
            utility_metrics=["link_forecasting"],
            temporal_metrics="auto",
            get_privacy_metric=False,
            embedder_path=f"checkpoints_lp/best_{dataset_name}.ckpt",
        )

        metrics = evaluator(
            original=reference_dataset,
            generated=generated_dataset,
        )

        result_folder = Path("generated_graphs") / "metrics"
        result_folder.mkdir(parents=True, exist_ok=True)

        # Save metrics as CSV
        dataframe = pd.Series(
            metrics,
            name=f"{model_name}-{dataset_name}{f'_{note}' if note != '' else ''}",
        )
        dataframe.to_csv(result_folder / f"{model_name}-{dataset_name}{f'_{note}' if note != '' else ''}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        required=False,
        metavar="--f",
        type=Path,
        nargs="+",
        help="The paths of the files to consider. If not given, all files in the generated_graphs folder are considered except those in the 'not-run' subfolder.",
        default=None,
    )
    parser.add_argument(
        "--test_file_path",
        required=False,
        metavar="--t",
        type=Path,
        help="The path to the test file to use as the reference dataset.",
        default=None,
    )

    args = vars(parser.parse_args())
    main(args["files"], args["test_file_path"])
