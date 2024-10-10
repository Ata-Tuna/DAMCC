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
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    README = f.read()


with open("LICENCE", "r", encoding="utf-8") as f:
    LICENSE = f.read()


setuptools.setup(
    name="brainiac-temporal",
    version="1.0.0",
    author="Euranova",
    author_email="research@euranova.eu",
    description="Official Brainiac Temporal library.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=("tests", "docs")),
    license=LICENSE,
    python_requires=">=3.8",
    zip_safe=False,
    # install_requires=_parse_requirements("requirements.txt"),
)
