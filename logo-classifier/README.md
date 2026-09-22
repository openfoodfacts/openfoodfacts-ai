# Logo Classifier

Machine learning classifier for detecting and categorizing logos in product images.

## Features

- Logo detection and classification
- Training pipeline for logo classification models
- ONNX model export and testing
- Dataset management tools
- Model evaluation and testing utilities

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

### Dataset Preparation
```bash
python dataset.py
```

### ONNX Model Testing
```bash
python test_saved_model_onnx.py
```

## Files

- `train.py` - Main training script for logo classification
- `dataset.py` - Dataset loading and preprocessing utilities
- `test_saved_model_onnx.py` - Testing utilities for ONNX exported models