MODEL_URL = "https://huggingface.co/openfoodfacts/crop-detection/resolve/main/weights/best_saved_model/best_float16.tflite?download=true"
MODEL_PATH = models/yolov8n_float16.tflite

.PHONY: *

init: hello install load-model

hello:
	@echo "🍋Welcome to the Crop Detection project.🍋"

install:
	@echo "Install dependencies."
	pip install -r requirements.txt

load-model:
	@echo "Load model from the HF repository 🤗: https://huggingface.co/openfoodfacts/crop-detection"
	@if [ ! -f "${MODEL_PATH}" ]; then \
		echo "Model not found. Downloading from HF repository 🤗..."; \
		wget -O "${MODEL_PATH}" "${MODEL_URL}" ; \
	else \
		echo "Model already exists in models/"; \
	fi