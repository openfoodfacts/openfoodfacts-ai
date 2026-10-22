# Language Identification

Machine learning models for automatic language identification in product text data.

## Features

- Language detection for product ingredients and descriptions
- Training pipelines for language classification models
- Data extraction and preprocessing scripts
- Model evaluation and metrics calculation
- Inference utilities for production use

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

Or install with pip:
```bash
pip install -r requirements.txt
```

## Usage

### Extract Data
```bash
poetry run python scripts/01_extract_data.py
```

### Calculate Metrics
```bash
poetry run python scripts/03_calculate_metrics.py
```

### Run Inference
```bash
poetry run python scripts/inference.py
```

## Project Structure

- `scripts/` - Data processing and model training scripts
  - `01_extract_data.py` - Data extraction from OpenFoodFacts
  - `03_calculate_metrics.py` - Model evaluation metrics
  - `inference.py` - Model inference utilities

## Dependencies

This project uses Poetry for dependency management. See `pyproject.toml` for the complete list of dependencies.