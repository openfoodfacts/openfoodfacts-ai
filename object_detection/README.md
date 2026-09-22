# Object Detection

Object detection models and tools for identifying various objects in food product images.

## Features

- Crop detection in product images
- TensorFlow Object Detection API integration
- YOLO model inference with TensorFlow Lite
- Dataset management and preprocessing
- Training pipelines for custom object detection models

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### YOLO TensorFlow Lite Inference
```bash
cd crop_detection/cli
python inference_yolo_tflite.py
```

### Run CLI Interface
```bash
cd crop_detection/cli
python -m __main__
```

## Structure

- `crop_detection/` - Crop detection specific models and tools
  - `cli/` - Command-line interface for inference
- `tensorflow_object_api/` - TensorFlow Object Detection API utilities
- `dataset/` - Dataset management tools
- `train/` - Training scripts and configurations