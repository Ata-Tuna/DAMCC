
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

from brainiac_temporal.data.utils import convert_insecta_file_to_tglib_file


def test_insecta_converter() -> None:
    """
    tests the insecta converter to tglib format
    """

    data = ["1 2 1 1\n", "1 3 4 2\n"]

    path = "sample"
    with open(path, "w") as file_out:
        file_contents = "".join(data)
        file_out.write(file_contents)

    convert_insecta_file_to_tglib_file(path, opposite_edges=False)

    with open(path + "-tglib", "r") as file:
        string_list = file.readlines()

    expected_data = ["1 2 1\n", "1 3 2\n"]
    assert string_list == expected_data
