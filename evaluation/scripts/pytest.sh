#!/bin/bash
set -ex

PYTHON="${PYTHON:=$(which python)}"

mkdir -p logs

$PYTHON -m pytest --pylint --mypy --cov=src --disable-pytest-warnings --durations=5 --cov-fail-under 95
# $PYTHON -m pytest -x --nbmake --overwrite "./examples"
