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
help:
	@cat Makefile

.EXPORT_ALL_VARIABLES:

# create an .env file to override the default settings
-include .env
export $(shell sed 's/=.*//' .env)


.PHONY: build docs

# ----------------
# default settings
# ----------------
# user
LOCAL_USER:=$(shell whoami)
LOCAL_USER_ID:=$(shell id -u)
# project
PROJECT_NAME:=brainiac-temporal
# python
PYTHON?=python
PYTHON_EXEC?=python -m
PYTHONVERSION?=3.8.13
VIRTUALENV_PATH=$(PYENV_ROOT)/versions/$(PROJECT_NAME)
PYTEST?=pytest
# poetry
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
POETRY=poetry
# docker
DOCKER?=docker
REGISTRY=registry.euranova.eu/rd/bishop/brainiac-1-temporal
SHM=4G
CPU="2"
MEMORY=32G
DOCKER_COMMON_FLAGS=--shm-size $(SHM) -m $(MEMORY) --cpus=$(CPU) --network=host --volume $(PWD):/workdir -e LOCAL_USER_ID -e LOCAL_USER
IMAGE=$(REGISTRY)/$(PROJECT_NAME)
IMAGE_PYTHON=/$(PROJECT_NAME)/bin/python
# You can set these variables from the command line, and also from the environment for the first two
SPHINXOPTS?=
SPHINXBUILD?=sphinx-build
SOURCEDIR=docs
BUILDDIR=$(SOURCEDIR)/_build
# others
PYTHON_EXEC_M1=OPENBLAS=${OPENBLAS} CFLAGS=${CFLAGS} LDFLAGS=${LDFLAGS} MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m
# cuda
CUDA_VERSION?=cu113
DEVICE=0

# -----------
# utilities
# -----------
size:
	du -hs * | sort -rh | head -10

chown:
	sudo chown $(LOCAL_USER) -R .

# -----------
# init
# -----------
install-pyenv:
	curl https://pyenv.run | bash

# correct python version and virtualenv using pyenv
python-install:
	pyenv install $(PYTHONVERSION)

virtualenv:
	pyenv virtualenv-delete $(PROJECT_NAME) || echo "no virtual env found, creating..."
	pyenv virtualenv $(PYTHONVERSION) $(PROJECT_NAME)

activate:
	pyenv shell $(PROJECT_NAME)

init: python-install virtualenv activate

# -----------
# install project's dependencies
# -----------
# dev dependencies
install-init:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade setuptools virtualenv wheel poetry

# delete poetry's cache
delete-poetry-cache:
	$(PYTHON_EXEC) $(POETRY) cache clear PyPI --all
	$(PYTHON_EXEC) $(POETRY) cache clear _default_cache --all

install-tglib: install-init 
	git clone https://gitlab.com/tgpublic/tglib.git scripts/tglib || echo "tglib repo detected, skipping clone"
	cd scripts/tglib/tglib_cpp \
    && (mkdir build-release \
    && cd build-release \
    && cmake .. -DCMAKE_BUILD_TYPE=Release && make) || echo ".so file already generated"
	mkdir $(VIRTUALENV_PATH)/lib/python3.8/site-packages/pytglib \
    && echo $(PWD)/scripts/tglib/tglib_cpp/build-release/src/python_binding/ > $(VIRTUALENV_PATH)/lib/python3.8/site-packages/pytglib.pth

# install main dependencies with poetry (dynamic installation)
install: install-tglib
	$(PYTHON_EXEC) $(POETRY) install --no-cache

# -----------
# testing
# -----------
# mypy
mypy:
	$(PYTHON_EXEC) mypy test

# pytest
pytest-src:
	$(PYTHON_EXEC) $(PYTEST) --testmon --mypy --pylint --all

pytest-nb:
	$(PYTHON_EXEC) $(PYTEST) --nbmake --overwrite "./notebooks"

pytest: pytest-src pytest-nb

# -----------
# git
# -----------
# delete all local branches that do not have a remote counterpart
git-clean:
	git remote prune origin
	git branch -r | awk '{print $$1}' | egrep -v -f /dev/fd/0 <(git branch -vv | grep origin) | awk '{print $$1}' | xargs git branch -D

# squash all commits before rebasing, see https://stackoverflow.com/questions/25356810/git-how-to-squash-all-commits-on-branch
git-squash:
	git reset $(git merge-base main $(git branch --show-current))
	git add -A
	git commit -m "squashed commit"

# -----------
# docs
# -----------
docs: SRC=src/brainiac_2
docs:
	@rm docs/brainiac_2.*
	@rm docs/modules.rst
	sphinx-apidoc -o docs $(SRC)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# -----------
# M1
# -----------
check-m1:
	@echo CFLAGS=${CFLAGS}
	@echo LDFLAGS=${LDFLAGS}
	@echo OPENBLAS=${OPENBLAS}
	$(PYTHON_EXEC) python -c 'import torch; print(torch.backends.mps.is_available())'
	$(PYTHON_EXEC) python -c 'import platform; print(platform.processor(), platform.system(), platform.architecture())'
	$(PYTHON_EXEC) python -c 'import pytorch_lightning as pl; trainer = pl.Trainer(accelerator="mps")'

# --- M1 MACS: currently struggling with torch >= 1.13
# MAIN: w poetry (dynamic)for M1 MACs
install-m1: PYTHON_EXEC=$(PYTHON_EXEC_M1)
install-m1: install-init
install-m1: install-w-poetry
install-m1: install-manual-pkgs
install-m1: install-project

add: PACK=
add:
	PYTHON_KEYRING_BACKEND=$(PYTHON_KEYRING_BACKEND) $(PYTHON_EXEC) $(POETRY) add $(PACK)


# MAIN: w pip (static) for M1 MACs
pip-install-m1: PYTHON_EXEC=$(PYTHON_EXEC_M1)
pip-install-m1: pip-install

# add packages on M1 MACs
add-m1: PYTHON_EXEC=OPENBLAS=${OPENBLAS} CFLAGS=${CFLAGS} LDFLAGS=${LDFLAGS} pyenv exec
add-m1: PACK=
add-m1:
	$(PYTHON_EXEC) poetry add $(PACK)

# -----------
# docker
# -----------
# build project's image
build: DOCKER_HOST=
build: BUILD_CMD=$(DOCKER) build -t $(REGISTRY)/$(PROJECT_NAME) --build-arg project_name=$(PROJECT_NAME) -f Dockerfile .
build:
	$(BUILD_CMD)

build-nohup: DOCKER_HOST=
build-nohup: build
build-nohup:
	mkdir -p logs
	nohup $(BUILD_CMD) 2>&1 > logs/build.log &

tag:
	$(DOCKER) image tag $(PROJECT_NAME):latest $(REGISTRY)/$(PROJECT_NAME):latest

push: DOCKER_HOST=
push:
	echo $(CI_JOB_TOKEN) | $(DOCKER) login -u $(LOCAL_USER) $(REGISTRY) --password-stdin
	$(DOCKER) image push $(REGISTRY)/$(PROJECT_NAME):latest

pull: DOCKER_HOST=
pull:
	echo $(CI_JOB_TOKEN) | $(DOCKER) login -u $(LOCAL_USER) $(REGISTRY) --password-stdin
	$(DOCKER) pull $(REGISTRY)/$(PROJECT_NAME):latest

# run pytest within a container
docker-pytest: DOCKER_HOST=
docker-pytest: DOCKER_FLAGS=
docker-pytest:
	$(DOCKER) run --rm $(DOCKER_FLAGS) \
		--name $(PROJECT_NAME)-pytest \
		$(DOCKER_COMMON_FLAGS) \
		-t $(REGISTRY)/$(PROJECT_NAME) $(IMAGE_PYTHON) -m pytest 2>&1 > pytest.log
	$(DOCKER) run --rm $(DOCKER_FLAGS) \
		--name $(PROJECT_NAME)-pytest \
		$(DOCKER_COMMON_FLAGS) \
		-t $(REGISTRY)/$(PROJECT_NAME) $(IMAGE_PYTHON) $(PYTEST) --nbmake --overwrite "./notebooks" 2>&1 >> pytest.log


# launch dev container
dev-container: DOCKER_HOST= 
dev-container:
	$(DOCKER) run --rm -d $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-dev \
		-t $(REGISTRY)/$(PROJECT_NAME) bash

# Jupyter

notebook-gpu: GPU_FLAGS=--runtime=nvidia --gpus all
notebook-gpu: notebook

notebook: DOCKER_HOST= 
notebook: PORTJUPY=8888
notebook:
	$(DOCKER) run --rm -it \
		--volume $(PWD):/workdir -e LOCAL_USER_ID -e LOCAL_USER \
		$(GPU_FLAGS) \
		--name $(PROJECT_NAME)-bash \
		-p $(PORTJUPY):8888 \
		-t $(REGISTRY)/$(PROJECT_NAME) jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

# launch bash session within a container
bash: DOCKER_HOST= 
bash: CONTAINER_NAME=
bash:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-bash-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) bash


bash-gpu: DOCKER_HOST= 
bash-gpu: CONTAINER_NAME=
bash-gpu:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--runtime=nvidia --gpus all \
		--name $(PROJECT_NAME)-bash-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) bash

# launch python session within a container
docker-python: DOCKER_HOST= 
docker-python:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-pytest \
		-t $(REGISTRY)/$(PROJECT_NAME) $(IMAGE_PYTHON)

# [DO NOT CALL THIS COMMAND]
run-base: CONFIG=gran.yaml
run-base: CONTAINER_NAME=$(shell bash -c 'echo $$RANDOM')
run-base: SCRIPT=/workdir/experiments/main.py
run-base: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
run-base: DOCKER_FLAGS=
run-base: OVERRIDE=
run-base: NOW=$(shell date '+%Y-%m-%d_%H:%M:%S')
run-base: CMD=$(IMAGE_PYTHON) -u $(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)
run-base:
	$(DOCKER) run --rm -it $(DOCKER_FLAGS) $(DOCKER_COMMON_FLAGS) \
		$(GPU_FLAGS) \
		--name $(PROJECT_NAME)-run-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		$(CMD)

# $(IMAGE_PYTHON) -u $(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)

# run: locally
run: DOCKER_HOST= 
run: run-base

# run-gpu: GPU_FLAGS=--runtime=nvidia --gpus all
# run-gpu: GPU_FLAGS=--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1
# run-gpu: GPU_FLAGS=--runtime=nvidia -e CUDA_VISIBLE_DEVICES=1
run-gpu: GPU_FLAGS=--runtime=nvidia --gpus '"device=$(DEVICE)"'
run-gpu: run-base

# run: will run remotely if the DOCKER_HOST variable is set correctly
# TODO: https://stackoverflow.com/questions/51305537/how-can-i-mount-a-volume-of-files-to-a-remote-docker-daemon
run-remote:
	$(DOCKER) volume create data-volume
	$(DOCKER) rm helper || echo "Container may not exist"
	$(DOCKER) create -v data-volume:/workdir --name helper busybox true
	$(DOCKER) cp $(PWD)/src helper:/workdir/src
	$(DOCKER) cp $(PWD)/scripts helper:/workdir/scripts
	$(DOCKER) cp $(PWD)/experiments helper:/workdir/experiments
	$(DOCKER) cp $(PWD)/configs helper:/workdir/configs
	$(DOCKER) rm helper
	$(DOCKER) run --rm -it -d \
		--network=host --volume data-volume:/workdir -e LOCAL_USER_ID -e LOCAL_USER \
		--name $(PROJECT_NAME)-train-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		$(IMAGE_PYTHON) -u $(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)

# check your docker information
docker-info:
	DOCKER_HOST=$(DOCKER_HOST) $(DOCKER) info

# Docker-compose: launch a jupyter notebook and tensorboard service
up: DOCKER_HOST=
up: PORT_JUPY=8888
up: PORT_TB=6006
up: UNIQUE=$(LOCAL_USER)-$(shell bash -c 'echo $$RANDOM')
up:
	$(DOCKER)-compose -p $(PROJECT_NAME)-$(UNIQUE) up -d

# tear down the notebook and tensorboard
down: DOCKER_HOST=
down: docker-compose.yml
	$(DOCKER)-compose down --volumes

# clean dangling images
clean: DOCKER_HOST=
clean:
	$(DOCKER) system prune -a

# WARNING: cleans everything, even images you may want to keep
clean-all: DOCKER_HOST=
clean-all:
	$(DOCKER) rmi $(docker images -f dangling=true -q)



