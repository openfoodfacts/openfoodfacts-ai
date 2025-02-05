# Train object detection models using Ultralytics library


# Export data in Ultralytics format

Ultralytics library expects data in a [specific format](https://docs.ultralytics.com/datasets/detect/).

From a Hugging Face dataset, you can export the data with the OFF ML CLI in `openfoodfacts-ai/ml_utils/labelstudio`:

```bash
python3 main.py export --from hf --to ultralytics --repo-id openfoodfacts/nutrition-table-detection --output-dir ~/datasets/nutrition-table-detection/ultralytics/
```

