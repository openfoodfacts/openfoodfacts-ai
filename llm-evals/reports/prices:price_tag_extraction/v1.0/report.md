
```bash
uv run main.py evaluate prices:price_tag_extraction --model gemini-3-pro-preview
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-8b-instruct
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-instruct
uv run main.py evaluate prices:price_tag_extraction --model openrouter:qwen/qwen3-vl-235b-a22b-thinking --max-concurrency 2
```

## Dataset details

Number of cases: 100
---
Tag counts in evaluated cases:
  country:fr: 70
  country:us: 13
  country:no: 11
  country:es: 5
  country:se: 1
  currency:eur: 75
  currency:usd: 13
  currency:nok: 11
  currency:sek: 1
  data-quality:barcode-unreadable: 10
  data-quality:price-truncated: 2
  data-quality:price-unreadable: 1
  type:product: 95
  type:category: 5

## Gemini 3 Pro Preview

Detailed scores:
  price: 98/98 (100.00% accuracy)
  barcode: 81/84 (96.43% accuracy)

## Google Gemini 2.5 Pro

Detailed scores:
  price: 96/98 (97.96% accuracy)
  barcode: 76/84 (90.48% accuracy)

## Google Gemini 2.5 Flash

Detailed scores:
  price: 91/98 (92.86% accuracy)
  barcode: 73/84 (86.90% accuracy)

## Google Gemini 2.5 Flash Lite

Detailed scores:
  price: 80/98 (81.63% accuracy)
  barcode: 72/84 (85.71% accuracy)

## OpenRouter Qwen 3 VL 8B Instruct

Detailed scores:
  price: 78/98 (79.59% accuracy)
  barcode: 54/84 (64.29% accuracy)

## OpenRouter Qwen 3 VL 8B Thinking

Detailed scores:
  price: 81/98 (82.65% accuracy)
  barcode: 56/84 (66.67% accuracy)

## OpenRouter Qwen 3 VL 235B A22B Instruct

Detailed scores:
  price: 85/98 (86.73% accuracy)
  barcode: 67/84 (79.76% accuracy)

# OpenRouter Qwen 3 VL 235B A22B Thinking

Detailed scores:
  price: 87/98 (88.78% accuracy)
  barcode: 62/84 (73.81% accuracy)

# OpenRouter Amazon Nova-2 Lite v1 (free)

Detailed scores:
  price: 67/98 (68.37% accuracy)
  barcode: 39/84 (46.43% accuracy)

## Summary table

|Model | Price Accuracy | Barcode Accuracy |
|-- | -- | -- |
|Gemini 3 Pro Preview | 100.00% | 96.43% |
|Google Gemini 2.5 Pro | 97.96% | 90.48% |
|Google Gemini 2.5 Flash |92.86% | 86.90% |
|Google Gemini 2.5 Flash Lite | 81.63% | 85.71% |
|OpenRouter Qwen 3 VL 8B Thinking |82.65% | 66.67% |
|OpenRouter Qwen 3 VL 8B Instruct |79.59% | 64.29% |
|OpenRouter Qwen 3 VL 235B A22B Thinking |88.78% | 73.81% |
|OpenRouter Qwen 3 VL 235B A22B Instruct |86.73% | 79.76% |
|OpenRouter Amazon Nova-2 Lite v1 (free) |68.37% | 46.43% |
