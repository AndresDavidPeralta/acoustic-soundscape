#  Makefile (Linux / macOS) - conda based

SHELL := /bin/bash

ENV_NAME := acustic-soundscape
MAIN := main.py
CONFIG_DIR := configs

.PHONY: help setup exp1 exp2 exp3 all clean

help:
	@echo
	@echo "Available commands:"
	@echo "  make setup     - create conda env from environment.yml"
	@echo "  make exp1      - run Experiment 1 (conda run)"
	@echo "  make exp2      - run Experiment 2 (conda run)"
	@echo "  make exp3      - run Experiment 3 (conda run)"
	@echo "  make all       - run all experiments sequentially"
	@echo "  make clean     - remove logs and caches"
	@echo

setup:
	@echo ">>> Creating conda environment '$(ENV_NAME)' from environment.yml"
	@conda env create -f environment.yml || echo "Environment may already exist; try: conda activate $(ENV_NAME)"

exp1:
	@echo ">>> Running Experiment 1"
	@conda run -n $(ENV_NAME) python $(MAIN) --config $(CONFIG_DIR)/experiment_1.yaml

exp2:
	@echo ">>> Running Experiment 2"
	@conda run -n $(ENV_NAME) python $(MAIN) --config $(CONFIG_DIR)/experiment_2.yaml

exp3:
	@echo ">>> Running Experiment 3"
	@conda run -n $(ENV_NAME) python $(MAIN) --config $(CONFIG_DIR)/experiment_3.yaml

all: exp1 exp2 exp3
	@echo ">>> All experiments completed successfully!"

clean:
	@echo "Cleaning logs and caches ..."
	@rm -rf results/logs __pycache__ .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
