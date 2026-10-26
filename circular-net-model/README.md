# Circular Model

Machine learning model for circular detection in product images, with barcode generation capabilities.

## Features

- Circular pattern detection in product images
- Barcode generation and processing
- Image dataset downloading from OpenFoodFacts
- Jupyter notebook with model training pipeline

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Download Images
```bash
python download_images.py
```

### Generate Barcodes
```bash
python generate_barcode.py
```

### Model Training
Open and run the `circular_model.ipynb` notebook for model training and evaluation.

## Files

- `circular_model.ipynb` - Main Jupyter notebook with model implementation
- `download_images.py` - Script to download images from OpenFoodFacts
- `generate_barcode.py` - Barcode generation utilities
- `images/` - Directory for storing downloaded images