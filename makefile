.PHONY: help
.PHONY: black black-check flake8
.PHONY: install install-dev install-devtools install-test install-lint
.PHONY: test
.PHONY: conda-env
.PHONY: black isort format
.PHONY: black-check isort-check format-check
.PHONY: flake8

.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "install"
	@echo "        Install backpack and dependencies"
	@echo "install-dev"
	@echo "        Install all development tools"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "conda-env"
	@echo "        Create conda environment 'backpack' with dev setup"
###
# Test coverage
test:
	@pytest -vx --cov=backpack .

###
# Linter and autoformatter

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

flake8:
	@flake8 .

isort:
	@isort --apply

isort-check:
	@isort --check

format:
	@make black
	@make isort
	@make black-check

format-check: black-check isort-check


###
# Installation

install:
	@pip install -r requirements.txt
	@pip install .

install-lint:
	@pip install -r requirements/lint.txt

install-test:
	@pip install -r requirements/test.txt

install-devtools:
	@echo "Install dev tools..."
	@pip install -r requirements-dev.txt

install-dev: install-devtools
	@echo "Install dependencies..."
	@pip install -r requirements.txt
	@echo "Uninstall existing version of backpack..."
	@pip uninstall backpack-for-pytorch
	@echo "Install backpack in editable mode..."
	@pip install -e .
	@echo "Install pre-commit hooks..."
	@pre-commit install

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml
