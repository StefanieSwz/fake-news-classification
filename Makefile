#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fake-news-classification
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 fakenews
	isort --check --diff --profile black fakenews
	black --check --config pyproject.toml fakenews

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml fakenews




## Set up python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) fakenews/data/make_dataset.py ${ARGS}

## Make Preprocess
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) fakenews/data/preprocessing.py ${ARGS}

## Make Preprocess
.PHONY: generate
generate:
	$(PYTHON_INTERPRETER) fakenews/data/data_generator.py ${ARGS}

## Make Train
.PHONY: train_model
train:
	$(PYTHON_INTERPRETER) fakenews/model/train_model.py ${ARGS}

## Make
.PHONY: wandb_registry
wandb_registry:
	$(PYTHON_INTERPRETER) fakenews/model/wandb_registry.py ${ARGS}

## Make
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) fakenews/model/predict.py ${ARGS}

## Make
.PHONY: transform_model
transform_model:
	$(PYTHON_INTERPRETER) fakenews/model/transform_model.py ${ARGS}


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
