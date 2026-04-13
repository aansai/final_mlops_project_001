.PHONY: help install lint format test data features train predict report docs clean

PYTHON := python
PIP    := pip

SRC       := Laptop_prediction       # ✅ matches your actual folder
DATA_PATH ?= data/raw/dataset.csv


help:
	@echo "Available commands:"
	@echo "  make install   - Install project + dev dependencies"
	@echo "  make lint      - Run flake8 linter"
	@echo "  make format    - Run black + isort formatter"
	@echo "  make test      - Run pytest with coverage"
	@echo "  make data      - Run dataset pipeline"
	@echo "  make features  - Run feature engineering"
	@echo "  make train     - Train the model"
	@echo "  make predict   - Run predictions"
	@echo "  make report    - Generate plots/reports"
	@echo "  make docs      - Serve mkdocs documentation"
	@echo "  make clean     - Remove cache and build files"


install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"         # ✅ quoted


lint:
	flake8 $(SRC) tests/


format:
	black $(SRC) tests/
	isort $(SRC) tests/


test:
	pytest tests/ -v --tb=short --cov=$(SRC) --cov-report=term-missing


data:
	$(PYTHON) -m $(SRC).dataset


features:
	$(PYTHON) -m $(SRC).features


train:
	$(PYTHON) -m $(SRC).modeling.train


predict:
	$(PYTHON) -m $(SRC).modeling.predict


report:
	$(PYTHON) -m $(SRC).plots


docs:
	mkdocs serve


# ✅ Windows-compatible clean
clean:
	del /S /Q *.pyc 2>nul
	for /d /r . %%d in (__pycache__)  do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (*.egg-info)   do @if exist "%%d" rd /s /q "%%d"