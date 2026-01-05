# Price tag evaluation

The evaluation was performed on the dataset with tag [llm-evals-price-tags-ds-1.2](https://github.com/openfoodfacts/openfoodfacts-ai/blob/llm-evals-price-tags-ds-1.2/llm-evals/llm_evals/tasks/prices/price_tag_extraction/dataset.yaml).


## Dataset characteristics

The dataset has the following characteristics:

```
---------------------------------------------
Number of cases: 474
Evaluated price extraction: 469
Evaluated barcode extraction: 409
---------------------------------------------
Tag counts in evaluated cases:
  country:it: 130
  country:es: 122
  country:de: 109
  country:fr: 77
  country:us: 13
  country:no: 11
  country:jp: 9
  country:se: 1
  country:nl: 1
  country:in: 1
  currency:eur: 432
  currency:usd: 13
  currency:nok: 11
  currency:jpy: 9
  currency:unknown: 8
  currency:sek: 1
  data-quality:barcode-unreadable: 31
  data-quality:not-a-price-tag: 4
  data-quality:barcode-truncated: 3
  data-quality:price-unreadable: 3
  data-quality:price-truncated: 2
  difficulty:barcode:easy: 18
  difficulty:barcode:hard: 9
  difficulty:barcode:medium: 5
  difficulty:price:hard: 4
  missing-barcode: 11
  missing-price: 1
  type:product: 454
  type:unknown: 8
  type:magazine: 6
  type:category: 5
  type:screenshot: 1
  with-fidelity-card-discount: 1
  with-internal-barcode: 35
  with-non-vat-price: 9
```

Here is a description of a few notable tags:

- country:*: The country associated with the proof.
- currency:*: The currency associated with the proof.
- data-quality:barcode-unreadable: The barcode is not readable on the price tag. We don't evaluate barcode extraction for these cases.
- data-quality:not-a-price-tag: The image is not a price tag. These cases are kept in the evaluation to see how models behave on such cases (we expect the model to not hallucinate a barcode).
- data-quality:barcode-truncated: The barcode is truncated on the price tag. We don't evaluate barcode extraction for these cases.
- data-quality:price-unreadable: The price is not readable on the price tag. We don't evaluate price extraction for these cases.
- data-quality:price-truncated: The price is truncated on the price tag. We don't evaluate price extraction for these cases.
- difficulty:barcode:*: The difficulty level of reading the barcode on the price tag (easy, medium, hard).
- difficulty:price:hard: The price is hard to read on the price tag.
- missing-barcode: The price tag is missing a barcode. We expect the model to not hallucinate a barcode for these cases.
- missing-price: The price tag is missing a price. We expect the model to not hallucinate a price for these cases.
- type:*: The type of price tag (product, category, magazine, screenshot).
- with-internal-barcode: The price tag contains an internal barcode (not the product EAN/UPC). We don't evaluate barcode extraction for these cases. It's indeed tricky for the model to distinguish between internal barcodes and product barcodes. We expect further post-processing to handle these cases (e.g. by checking the extracted barcode against OFF).
- with-non-vat-price: The price tag contains a non-VAT price. We expect the model to extract both the non-VAT price and the VAT price if present, so that `selected_price` method returns the correct price.

## Summary of results

| Model                                      | Price Accuracy    | Barcode Accuracy |
|--------------------------------------------|-------------------|------------------|
| Gemini 3 Pro Preview (high thinking)       | 99.57%            | 98.78%           |
| Gemini 3 flash-preview native minimal thinking | 99.36%        | 97.07%           |
| Gemini 3 flash-preview native thinking     | 99.15%            | 97.80%           |
| Gemini 2.5-pro native thinking             | 99.15%            |  97.31%          |
| Gemini 2.5-pro native minimal thinking     | 98.29%            | 97.80%           |
| Gemini 2.5-flash-preview-09-2025 native no thinking | 97.23%   | 95.60%           |
| Gemini 2.5-flash native thinking            | 96.59%           | 95.35%           |
| Gemini 2.5-flash native no thinking        | 96.16%            | 93.89%           |
| Gemini 2.5-flash-lite-preview-09-2025 native no thinking  | 95.95%  | 91.20%      |
| OpenRouter Qwen 3 VL 235B A22B Instruct | 95.74%               | 90.22%           |
| OpenRouter Qwen 3 VL 235B A22B Thinking   | 95.31%             | 88.75%           |
| OpenRouter Qwen 3 VL 8B Thinking          | 93.60%             | 80.68%           |
| Z.AI GLM 4.6V (106B)                      | 92.11%             | 82.64%           |
| Gemini 2.5-flash-lite native no thinking   | 90.62%            | 86.06%           |
| OpenRouter Qwen 3 VL 8B Instruct          | 89.13%             | 83.86%           |
| Gemma 3 27B instruct                       | 81.88%            | 71.64%           |
| Mistral Ministral 14B 2512 Instruct       | 79.74%             | 76.77%           |
| Gemma 3 12B instruct                       | 44.56%            | 68.95%           |

## Model details

Below are the command run for each evaluation run along with the evaluation results. Details of individual case evaluation can be obtained by running the command again.

### Gemini 3 Pro Preview (native high thinking - default)

```bash
GOOGLE_GENAI_USE_VERTEXAI='false' uv run main.py evaluate --task prices:price_tag_extraction --model google-gla:gemini-3-pro-preview --max-concurrency 2 --thinking-config HIGH --output-mode native
```

```
Detailed scores:
  price: 467/469 (99.57% accuracy)
  barcode: 404/409 (98.78% accuracy)
```

### Gemini 3 flash preview native thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-3-flash-preview --max-concurrency 5 --output-mode native
```

```
Detailed scores:
  price: 465/469 (99.15% accuracy)
  barcode: 400/409 (97.80% accuracy)
```

### Gemini 3 flash preview native minimal thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-3-flash-preview --max-concurrency 5 --output-mode native --thinking-config MINIMAL
```

```
Detailed scores:
  price: 466/469 (99.36% accuracy)
  barcode: 397/409 (97.07% accuracy)
```

### Gemini 2.5-pro native thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-2.5-pro --max-concurrency 5 --output-mode native
```

```
Detailed scores:
  price: 465/469 (99.15% accuracy)
  barcode: 398/409 (97.31% accuracy)
```

### Gemini 2.5-pro native minimal thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-2.5-pro --max-concurrency 5 --output-mode native --thinking-config 128
```

```
Detailed scores:
  price: 461/469 (98.29% accuracy)
  barcode: 400/409 (97.80% accuracy)
```

### Gemini 2.5-flash native thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-2.5-flash --max-concurrency 5
```

```
Detailed scores:
  price: 453/469 (96.59% accuracy)
  barcode: 390/409 (95.35% accuracy)
```

### Gemini 2.5-flash native no thinking

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-2.5-flash --max-concurrency 5 --output-mode native --thinking-config 0
```

```
Detailed scores:
  price: 451/469 (96.16% accuracy)
  barcode: 384/409 (93.89% accuracy)
```

### Gemini 2.5-flash-preview-09-2025 native no thinking

```bash
GOOGLE_GENAI_USE_VERTEXAI='false' uv run main.py evaluate --task prices:price_tag_extraction --model google-gla:gemini-2.5-flash-preview-09-2025 --max-concurrency 5 --output-mode native --thinking-config 0
```

```
Detailed scores:
  price: 456/469 (97.23% accuracy)
  barcode: 391/409 (95.60% accuracy)
```

### Gemini 2.5-flash-lite native no thinking (default)

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model google-vertex:gemini-2.5-flash-lite --max-concurrency 5 --output-mode native --thinking-config 0
```

```
Detailed scores:
  price: 425/469 (90.62% accuracy)
  barcode: 352/409 (86.06% accuracy)
```

### Gemini 2.5-flash-lite-preview-09-2025 native no thinking (default)

```bash
GOOGLE_GENAI_USE_VERTEXAI='false' uv run main.py evaluate --task prices:price_tag_extraction --model google-gla:gemini-2.5-flash-lite-preview-09-2025 --max-concurrency 5 --output-mode native --thinking-config 0
```

```
Detailed scores:
  price: 450/469 (95.95% accuracy)
  barcode: 373/409 (91.20% accuracy)
```

### OpenRouter Qwen 3 VL 235B A22B Instruct

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-instruct --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 449/469 (95.74% accuracy)
  barcode: 369/409 (90.22% accuracy)
```

### OpenRouter Qwen 3 VL 235B A22B Thinking (default thinking)

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-thinking --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 447/469 (95.31% accuracy)
  barcode: 363/409 (88.75% accuracy)
```

### OpenRouter Qwen 3 VL 8B Instruct

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-instruct --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 418/469 (89.13% accuracy)
  barcode: 343/409 (83.86% accuracy)
```

### OpenRouter Qwen 3 VL 8B Thinking (default thinking)

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-thinking --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 439/469 (93.60% accuracy)
  barcode: 330/409 (80.68% accuracy)
```

### Gemma 3 27B instruct

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:google/gemma-3-27b-it --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 384/469 (81.88% accuracy)
  barcode: 293/409 (71.64% accuracy)
```

### Gemma 3 12B instruct

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:google/gemma-3-12b-it --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 209/469 (44.56% accuracy)
  barcode: 282/409 (68.95% accuracy)
```

### Z.AI GLM 4.6V (106B)

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:z-ai/glm-4.6v --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 432/469 (92.11% accuracy)
  barcode: 338/409 (82.64% accuracy)
```

### Mistral Ministral 14B 2512 Instruct

```bash
uv run main.py evaluate --task prices:price_tag_extraction --model openrouter:mistralai/ministral-14b-2512 --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 374/469 (79.74% accuracy)
  barcode: 314/409 (76.77% accuracy)
```