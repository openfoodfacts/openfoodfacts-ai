# Model Overview

This document provides an overview of the models used in the project, including their configurations and performance metrics.

## Model Description

The models used in this project are based on the YOLO (You Only Look Once) architecture, which is a state-of-the-art object detection model. The models have been trained on a custom dataset of price tags to detect and annotate price tags in images.

## Model Tested

| Base Model | Resolution | mAP val |
|------------|------------|---------|
| YOLO11n    | 640x640    | 0.78*   |
| YOLO11x    | 640x640    | 0.74    |
| YOLO11x    | 960x960    | 0.76    |

* This was trained on a dataset with fewer fruits and vegetables pictures, so the mAP is higher.
