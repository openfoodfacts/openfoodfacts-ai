# Price Tag Detection

Machine learning models for detecting and extracting price information from product price tags.

## Features

- Price tag detection in product images
- YOLO model training and inference
- Label Studio integration for data annotation
- Jupyter notebooks for data exploration and model development
- Utility scripts for data processing and model evaluation

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Load Data to Label Studio
Open and run:
```bash
jupyter notebook notebooks/1_load_data_to_ls.ipynb
```

### Update Label Studio Tasks
```bash
jupyter notebook notebooks/opt_update_ls_tasks.ipynb
```

### Run YOLO Model
```bash
python notebooks/3_run_yolo_model.py
```

## Structure

- `notebooks/` - Jupyter notebooks for data processing and model development
- `models/` - Trained models and model configurations
- `utility_scripts/` - Helper scripts for data processing and evaluation

## Files

- `1_load_data_to_ls.ipynb` - Notebook for loading data into Label Studio
- `opt_update_ls_tasks.ipynb` - Notebook for updating Label Studio annotation tasks
- `3_run_yolo_model.py` - Script for running YOLO model inference