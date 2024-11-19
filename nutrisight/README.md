# NutriSight

## Export

To export the model to ONNX format, create a virtualenv with the following dependencies:

```bash
pip install onnx==1.16.2 onnxruntime==1.19.2 torch==2.4.1+cpu transformers==4.44.2 optimum==1.22.0
```

Then run the following command:

```bash
optimum-cli export onnx -m openfoodfacts/nutrition-extractor --opset 19 --task token-classification nutrition-extractor-onnx-19
```
