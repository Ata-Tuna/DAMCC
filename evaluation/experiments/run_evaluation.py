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
from typing import List, Optional
import pickle

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
    "age": True,
    "damnets": True,
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
]


def check_files_are_valid(files: List[Path]) -> None:
    """
    Checks that all the files in the list respect the naming convention and that for update models a test sequence is given with the generated sequence.
    """
    for file in files:
        extensionless_name = file.stem

        name_, _ = extensionless_name, ""
        if "_" in extensionless_name:
            name_, _ = extensionless_name.split("_")
        parts = name_.split("-")

        if extensionless_name in ["metrics", "not-run"]:
            continue

        if len(parts) != 2:
            raise ValueError(
                f"Wrong file name. File name (without extension) must be '[model]-[dataset].[extension]' for generated graphs."
            )

        model = parts[0]
        dataset = parts[1]

        if model not in available_models:
            raise ValueError(
                f"{model} not in available models. File name (without extension) must be '[model]-[dataset]-test' for test graphs or '[model]-[dataset]' for graphs."
            )
        if dataset not in available_datasets:
            raise ValueError(
                f"{dataset} not in available datasets. File name (without extension) must be '[model]-[dataset]-test' for test graphs or '[model]-[dataset]-gen' for generated graphs."
            )


def main(
    files: Optional[List[Path]] = None,
    reference_dataset_path: Optional[Path] = None
) -> None:
    directory = Path("generated_graphs")

    if files is None:
        files = [x for x in directory.iterdir()]

    check_files_are_valid(files)

    # Load the reference dataset from the provided path
    if reference_dataset_path is None:
        raise ValueError("Reference dataset path must be provided.")
    
    with open(reference_dataset_path, 'rb') as file:
        reference_dataset = pickle.load(file)

    for file in files:
        extensionless_name = file.stem
        name_, note = extensionless_name, ""
        if "_" in extensionless_name:
            name_, note = extensionless_name.split("_")
        parts = name_.split("-")

        if extensionless_name in ["metrics", "not-run"]:
            continue

        model_name, dataset_name = parts

        if model_name == "damnets" or model_name == "age":
            converter = convert_damnets_sample_to_tgt
        elif model_name == "dymond":
            converter = convert_dymond_sample_to_tgt
        elif model_name == "d2g2":
            converter = convert_d2g2_sample_to_tgt
        elif model_name == "tigger":
            converter = convert_tigger_sample_to_tgt
        else:
            raise ValueError(f"Unknown model {model_name}")

        generated_dataset = converter(file)       

        evaluator = MetricEvaluator(
            statistics="all",
            temporal_aggregation="dtw",
            utility_metrics=["link_forecasting"],
            temporal_metrics="auto",
            get_privacy_metric=False,
            embedder_path=f"checkpoints_lp/best_{dataset_name}.ckpt",
        )

        print(reference_dataset)
        print(generated_dataset)

        # Define the folder path where you want to save the file
        folder_path = 'for_me'

        # Ensure the folder exists, and if not, create it
        os.makedirs(folder_path, exist_ok=True)

        # Define the full file path (folder + filename)
        file_path = os.path.join(folder_path, 'generated_dataset.pkl')
        # Save the object to the file
        with open(file_path, 'wb') as file:
            pickle.dump(generated_dataset, file)

        print(f"Object saved as '{file_path}'")

        metrics = evaluator(
            original=reference_dataset,
            generated=generated_dataset,
        )

        result_folder = os.path.join("generated_graphs", "metrics")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        dataframe = pd.Series(
            metrics,
            name=f"{model_name}-{dataset_name}{f'_{note}' if note != '' else ''}",
        )
        dataframe.to_csv(
            os.path.join(
                result_folder,
                f"{model_name}-{dataset_name}{f'_{note}' if note != '' else ''}.csv",
            )
        )


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
        "--reference_dataset",
        required=True,
        metavar="--rd",
        type=Path,
        help="The path to the reference dataset pickle file.",
    )

    args = vars(parser.parse_args())
    main(args["files"], args["reference_dataset"])
