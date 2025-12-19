# Price tag evaluation report


Get existing price tag IDs:

```bash
yq e -o=json dataset.yaml | jq '.cases[] | .metadata.price_tag_id' > existing_price_tag_ids.txt
```

or:

```bash
yq e -o=json dataset.yaml | jq '.cases[] | .metadata.price_tag_id' > ~/Projects/openfoodfacts-ai/open-prices/price_tag_extraction/existing_price_tag_ids.txt
```

## Gemini 3 Pro Preview (native high thinking - default)

```bash
GOOGLE_GENAI_USE_VERTEXAI='false' uv run main.py evaluate prices:price_tag_extraction --model google-gla:gemini-3-pro-preview --max-concurrency 2 --thinking-config HIGH --output-mode native --output-path reports/gemini-3-pro-preview-high-think-native-output.json
```

Detailed scores:
  price: 95/98 (96.94% accuracy)
  barcode: 80/84 (95.24% accuracy)


## Gemini 2.5-pro native thinking

```bash
uv run main.py evaluate prices:price_tag_extraction --model google-vertex:gemini-2.5-pro --max-concurrency 5 --output-mode native --output-path reports/gemini-2.5-pro-default-think-native-output.json
```

Detailed scores:
  price: 94/98 (95.92% accuracy)
  barcode: 75/84 (89.29% accuracy)

## Gemini 2.5-pro native minimal thinking

```bash
uv run main.py evaluate prices:price_tag_extraction --model google-vertex:gemini-2.5-pro --max-concurrency 5 --output-mode native --thinking-config 128 --output-path reports/gemini-2.5-pro-minimal-thinking-native-output.json
```

Detailed scores:
  price: 94/98 (95.92% accuracy)
  barcode: 75/84 (89.29% accuracy)

## Gemini 2.5-flash native thinking

```bash
uv run main.py evaluate prices:price_tag_extraction --model google-vertex:gemini-2.5-flash --max-concurrency 5 --output-mode native --output-path reports/gemini-2.5-flash-default-think-native-output.json
```

Detailed scores:
  price: 89/98 (90.82% accuracy)
  barcode: 76/84 (90.48% accuracy)

## Gemini 2.5-flash native no thinking

```bash
uv run main.py evaluate prices:price_tag_extraction --model google-vertex:gemini-2.5-flash --max-concurrency 5 --output-mode native --thinking-config 0 --output-path reports/gemini-2.5-flash-no-think-native-output.json
```

Detailed scores:
  price: 92/98 (93.88% accuracy)
  barcode: 72/84 (85.71% accuracy)

## Gemini 2.5-flash-lite native no thinking (default)

```bash
uv run main.py evaluate prices:price_tag_extraction --model google-vertex:gemini-2.5-flash-lite --max-concurrency 5 --output-mode native --thinking-config 0 --output-path reports/gemini-2.5-flash-lite-no-think-native-output.json
```

Detailed scores:
  price: 86/98 (87.76% accuracy)
  barcode: 63/84 (75.00% accuracy)


## OpenRouter Qwen 3 VL 235B A22B Instruct

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-instruct --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 90/98 (91.84% accuracy)
  barcode: 64/84 (76.19% accuracy)


## OpenRouter Qwen 3 VL 235B A22B Thinking (default thinking)

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-thinking --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 87/98 (88.78% accuracy)
  barcode: 61/84 (72.62% accuracy)

## OpenRouter Qwen 3 VL 8B Instruct

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-instruct --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 77/98 (78.57% accuracy)
  barcode: 52/84 (61.90% accuracy)


## OpenRouter Qwen 3 VL 8B Thinking (default thinking)

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-thinking --max-concurrency 2 --output-mode 'native+prompted'
```


## Gemma 3 27B instruct

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:google/gemma-3-27b-it --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 74/98 (75.51% accuracy)
  barcode: 50/84 (59.52% accuracy)


## Gemma 3 12B instruct

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:google/gemma-3-12b-it --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 53/98 (54.08% accuracy)
  barcode: 54/84 (64.29% accuracy)

## Z.AI GLM 4.6V

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:z-ai/glm-4.6v --max-concurrency 2 --output-mode 'native+prompted'
```

Output was invalid.

# Mistral Ministral 14B 2512 Instruct

```bash
uv run main.py evaluate prices:price_tag_extraction --model openrouter:mistralai/ministral-14b-2512 --max-concurrency 2 --output-mode 'native+prompted'
```

Detailed scores:
  price: 80/98 (81.63% accuracy)
  barcode: 57/84 (67.86% accuracy)

## Summary of results

|Model | Price Accuracy | Barcode Accuracy |
|-- | -- | -- |
|Gemini 3 Pro Preview  (thinking high - default) | 96.94% | 95.24% |
|Google Gemini 2.5 Pro (default thinking) | 95.92% | 89.29% |
|Google Gemini 2.5 Pro (minimal thinking) | 95.92% | 89.29% |
|Google Gemini 2.5 Flash (default thinking) | 90.82% | 90.48% |
|Google Gemini 2.5 Flash Lite (no thinking - default) | 87.76% | 75.00% |
| Gemma 3 27B Instruct | 75.51% | 59.52% |
| Gemma 3 12B Instruct | 54.08% | 64.29% |
|OpenRouter Qwen 3 VL 8B Thinking (default thinking) | 84.38% | 67.07% |
|OpenRouter Qwen 3 VL 8B Instruct | 78.57% | 61.90% |
|OpenRouter Qwen 3 VL 235B A22B Thinking (default thinking) | 88.78% | 72.62% |
|OpenRouter Qwen 3 VL 235B A22B Instruct | 91.84% | 76.19% |
|OpenRouter Mistral Ministral 14B 2512 Instruct | 81.63% | 67.86% |
