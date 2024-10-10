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
FROM python:3.8.13
# FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG project_name=brainiac-temporal

ENV PROJECT_NAME=$project_name

# Create workdir and copy dependency files
RUN mkdir -p /workdir
COPY pyproject.toml /workdir/pyproject.toml
COPY Makefile /workdir/Makefile
COPY scripts /workdir/scripts
COPY README.md /workdir/README.md
COPY LICENCE /workdir/LICENCE
COPY src /workdir/src
COPY poetry.lock /workdir/

# Install system packages
RUN umask 022 && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils ca-certificates gosu sudo cmake \
    && rm -rf /var/lib/apt/lists/*

# Change shell to be able to easily activate virtualenv
SHELL ["/bin/bash", "-c"]
WORKDIR /workdir

# Install project
RUN umask 022 && apt-get update \
    # Install system packages
    && apt-get install -y --no-install-recommends apt-utils ca-certificates gosu sudo git \
    && rm -rf /var/lib/apt/lists/* \
    # Install Python dependencies
    && pip install virtualenv \
    && virtualenv /$PROJECT_NAME \
    && source /$PROJECT_NAME/bin/activate \
    && make install VIRTUALENV_PATH=/$PROJECT_NAME \
    && cp poetry.lock /tmp/. \
    && rm -r /root/.cache \
    && chmod -R 777 /$PROJECT_NAME

# Handle user-permissions using GOSU (https://denibertovic.com/posts/handling-permissions-with-docker-volumes/): The entrypoint script `entrypoint.sh` is needed to log you in within the container at runtime: this means that any file you create in the container will belong to your user ID, not to root's, thus solving all those annoying permission-related issues
COPY scripts /workdir/scripts
RUN chmod -R 777 /workdir

# TensorBoard
EXPOSE 6006
# Jupyter Notebook
EXPOSE 8888

# Set entrypoint and default container command
ENTRYPOINT ["/workdir/scripts/entrypoint.sh"]
