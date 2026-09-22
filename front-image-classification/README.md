# Front Image Classification

Image classification system for categorizing front-facing product images using machine learning.

## Features

- Product front image classification
- Training pipeline with data augmentation
- CLI interface for training and inference
- Integration with OpenFoodFacts database
- Support for multiple ML backends

## Installation

This project uses Python script dependencies (PEP 723). The dependencies are defined inline in the script files.

For manual installation:

```bash
pip install typer tqdm Pillow ultralytics albumentations opencv-python numpy openfoodfacts duckdb torch
```

## Usage

### Training
```bash
python train.py
```

### CLI Interface
```bash
python cli.py --help
```

## Files

- `train.py` - Main training script with inline dependencies
- `cli.py` - Command-line interface for the classifier
- `ml_commons.py` - Common ML utilities and data transformations