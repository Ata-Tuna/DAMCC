
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

# pylint: disable=unused-import
import os

import pyrootutils  # type: ignore
import pytest

ROOT = pyrootutils.find_root(
    search_from=os.path.dirname(__file__),
    indicator=[".git", "pyproject.toml", "setup.py"],
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Define CL args."""
    parser.addoption("--all", action="store_true", default=False)
    parser.addoption("--use-gpu", action="store_true", default=False)


@pytest.fixture(scope="session")
def all_tests(request: pytest.FixtureRequest) -> bool:
    """Whether to perform all tests or not."""
    all_tests_var = request.config.getoption("--all")
    return bool(all_tests_var)


@pytest.fixture(scope="session")
def use_gpu(request: pytest.FixtureRequest) -> bool:
    """Whether to test on GPU or not."""
    use_gpu_var: bool = request.config.getoption("--use-gpu")
    return use_gpu_var


@pytest.fixture(scope="session")
def data_path() -> str:
    """Path where to download any dataset."""
    return os.environ.get("PATH_DATASETS", "./.data")


@pytest.fixture(scope="session")
def checkpoints_path() -> str:
    """Path where to store checkpoints."""
    return os.environ.get("PATH_CHECKPOINTS", "./checkpoints")
