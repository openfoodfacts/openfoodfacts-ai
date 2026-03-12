# Price tag evaluation

The evaluation was performed on the dataset with tag [llm-evals-price-tags-ds-2.0](https://github.com/openfoodfacts/openfoodfacts-ai/blob/llm-evals-price-tags-ds-2.0/llm-evals/llm_evals/tasks/prices/price_tag_extraction/dataset.yaml).


## Dataset characteristics

The dataset has the following characteristics:

---------------------------------------------
Number of cases: 528
Evaluated price extraction: 523
Evaluated barcode extraction: 408
---------------------------------------------
Tag counts in evaluated cases:
  country:fr: 131
  country:it: 130
  country:es: 122
  country:de: 109
  country:us: 13
  country:no: 11
  country:jp: 9
  country:se: 1
  country:nl: 1
  country:in: 1
  currency:eur: 481
  currency:usd: 13
  currency:unknown: 13
  currency:nok: 11
  currency:jpy: 9
  currency:sek: 1
  data-quality:barcode-unreadable: 31
  data-quality:not-a-price-tag: 5
  data-quality:barcode-truncated: 3
  data-quality:price-unreadable: 3
  data-quality:price-truncated: 2
  data-quality:category-truncated: 2
  data-quality:category-unreadable: 1
  difficulty:barcode:easy: 18
  difficulty:barcode:hard: 9
  difficulty:barcode:medium: 5
  difficulty:price:hard: 4
  missing-barcode: 11
  missing-price: 1
  type:product: 454
  type:category: 55
  type:unknown: 12
  type:magazine: 6
  type:screenshot: 1
  with-fidelity-card-discount: 1
  with-internal-barcode: 35
  with-non-vat-price: 9


### Gemini 3 flash preview native minimal thinking

```bash
uv run main.py evaluate from-api --task prices:price_tag_extraction --model google-vertex:gemini-3-flash-preview --max-concurrency 5 --output-mode native --thinking-config MINIMAL
```

```
Detailed scores:
  price: 518/523 (99.04% accuracy)
  barcode: 397/408 (97.30% accuracy)
  uncertain_barcode_or_product_name: 492/528 (93.18% accuracy)
  category: 48/57 (84.21% accuracy)
```

### OpenRouter Qwen3.5 397B A17B

```bash
uv run main.py evaluate from-api --task prices:price_tag_extraction --model openrouter:qwen/qwen3.5-397b-a17b --max-concurrency 2 --output-mode 'native+prompted'
```

### OpenRouter Qwen 3 VL 8B Instruct

```bash
uv run main.py evaluate from-api --task prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-instruct --max-concurrency 2 --output-mode 'native+prompted'
```

```
Detailed scores:
  price: 466/523 (89.10% accuracy)
  barcode: 345/408 (84.56% accuracy)
  uncertain_barcode_or_product_name: 497/528 (94.13% accuracy)
  category: 19/66 (28.79% accuracy)
```


### OpenRouter Molmo 8B

```bash
uv run main.py evaluate from-api --task prices:price_tag_extraction --model openrouter:allenai/molmo-2-8b:free --max-concurrency 2 --output-mode 'native+prompted'
```


## Fine-tuned models


### Qwen3 VL 8B Instruct

#### rank 16

##### with constraints

```bash
uv run main.py evaluate from-prediction-file --task prices:price_tag_extraction --hf-repo-id openfoodfacts/price-tag-extractor --revision 2026-01-13-qwen3-vl-8b-lora-r-16 --hf-prediction-path predictions/val.jsonl
```

Detailed scores:
  price: 501/523 (95.79% accuracy)
  barcode: 359/408 (87.99% accuracy)
  uncertain_barcode_or_product_name: 488/528 (92.42% accuracy)
  category: 3/59 (5.08% accuracy)

##### without constraints

```bash
uv run main.py evaluate from-prediction-file --task prices:price_tag_extraction --hf-repo-id openfoodfacts/price-tag-extractor --revision 2026-01-13-qwen3-vl-8b-lora-r-16 --hf-prediction-path predictions/val_no_constraints.jsonl
```

Detailed scores:
  price: 501/522 (95.98% accuracy)
  barcode: 357/408 (87.50% accuracy)
  uncertain_barcode_or_product_name: 486/527 (92.22% accuracy)
  category: 3/63 (4.76% accuracy)


#### rank 32 


##### with constraints

```bash
uv run main.py evaluate from-prediction-file --task prices:price_tag_extraction --hf-repo-id openfoodfacts/price-tag-extractor --revision 2026-01-23-qwen3-vl-8b-lora-r-32 --hf-prediction-path predictions/val.jsonl
```

Detailed scores:
  price: 500/523 (95.60% accuracy)
  barcode: 360/408 (88.24% accuracy)
  uncertain_barcode_or_product_name: 491/528 (92.99% accuracy)
  category: 34/59 (57.63% accuracy)

##### without constraints

```bash
uv run main.py evaluate from-prediction-file --task prices:price_tag_extraction --hf-repo-id openfoodfacts/price-tag-extractor --revision 2026-01-23-qwen3-vl-8b-lora-r-32 --hf-prediction-path predictions/val_no_constraints.jsonl
```

Detailed scores:
  price: 492/510 (96.47% accuracy)
  barcode: 361/408 (88.48% accuracy)
  uncertain_barcode_or_product_name: 477/515 (92.62% accuracy)
  category: 22/46 (47.83% accuracy)