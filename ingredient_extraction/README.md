# Ingredient Extraction

Machine learning models and tools for extracting structured ingredient information from product text.

## Features

- Dataset generation for ingredient extraction tasks
- Model training and fine-tuning pipelines
- LayoutLM-based document understanding
- Model analysis and evaluation tools
- Streamlit demo interface

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Generation
```bash
cd dataset-generation
python generate_dataset.py
```

### Model Training
```bash
cd train
python train_model.py
```

### LayoutLM Training
```bash
cd train-layoutlm
python train_layoutlm.py
```

### Model Analysis
```bash
cd model-analysis
python evaluate_model.py
```

### Demo
```bash
streamlit run model-analysis/streamlit_demo.py
```

## Structure

- `dataset-generation/` - Scripts for creating training datasets
- `train/` - Standard model training pipeline
- `train-layoutlm/` - LayoutLM-specific training code
- `model-analysis/` - Model evaluation and analysis tools