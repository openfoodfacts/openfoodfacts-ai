# Price tag extraction

This report outlines the methodology and results of training a visual language model to extract price tag data from images, as part of the NLNet NGI0 project.

## Evaluation benchmark

An evaluation benchmark was created using data from the Open Prices database. For each detected price tag, user confirmed or corrected the price, the product barcode or category, and discount information.
About 500 samples were collected for evaluation purposes. A high performance model (Gemini 3 Pro Preview) was run on the evaluation benchmark, which helped to identify and correct errors in the benchmark data.

## Training dataset collection

Ensuring diversity in the training dataset is crucial for training a model robust to various real-world price tags. Country diversity was prioritized, as country of origin significantly influences price tag formats and styles.

The country distribution is as follows:

```sql
SELECT
  locations.osm_address_country_code,
  COUNT(*) AS count
FROM
  price_tags
  JOIN proofs ON price_tags.proof_id = proofs.id
  JOIN locations ON proofs.location_id = locations.id
WHERE
  price_tags.price_id IS NOT NULL
GROUP BY
  locations.osm_address_country_code
ORDER BY
  count DESC;
```

| osm_address_country_code | count |
| ------------------------ | ----- |
| FR                       | 50893 |
| NO                       | 9742  |
| US                       | 8013  |
| DE                       | 2395  |
| FI                       | 986   |
| BE                       | 758   |
| JP                       | 526   |
| GB                       | 384   |
| IT                       | 354   |
| ES                       | 322   |
| RU                       | 290   |
| EE                       | 287   |
| SE                       | 279   |
| CH                       | 251   |
| PL                       | 197   |
| TW                       | 162   |
| HR                       | 98    |
| LT                       | 95    |
| null                     | 77    |
| CA                       | 66    |
| KZ                       | 66    |
| IN                       | 57    |
| NL                       | 53    |
| TN                       | 48    |
| AT                       | 46    |
| TH                       | 38    |
| LA                       | 33    |
| PT                       | 26    |
| MY                       | 25    |
| MA                       | 10    |
| DK                       | 6     |
| AU                       | 6     |
| GR                       | 6     |
| PH                       | 5     |
| CZ                       | 5     |
| AL                       | 5     |
| BD                       | 5     |
| BR                       | 5     |
| IE                       | 4     |
| PK                       | 4     |
| AE                       | 4     |
| SK                       | 4     |
| AR                       | 4     |
| RO                       | 3     |
| HU                       | 3     |
| KR                       | 3     |
| QA                       | 2     |
| PE                       | 2     |
| DZ                       | 2     |
| UA                       | 2     |
| ZA                       | 2     |
| BG                       | 2     |
| LU                       | 2     |
| NZ                       | 2     |
| SG                       | 1     |
| CO                       | 1     |
| SN                       | 1     |
| CY                       | 1     |
| KE                       | 1     |
| TR                       | 1     |
| GA                       | 1     |
| PS                       | 1     |
| CK                       | 1     |
| SA                       | 1     |
| CN                       | 1     |

The country distribution shows a heavy skew towards France, so we limited the number of samples from France in the training dataset to ensure better representation of other countries.

We need the image URL for each price tag image, but it's not stored directly in the database. The `get_price_tag_image_path` function was created to generate the full image path for price tag images based on their ID:

```sql
CREATE OR REPLACE FUNCTION get_price_tag_image_path(p_price_tag_id integer)
RETURNS text
LANGUAGE sql
AS $$
    WITH padded AS (
        SELECT lpad(p_price_tag_id::text, 9, '0') AS id_str
    ),
    parts AS (
        SELECT
            id_str,
            substr(id_str, 1, 3) AS part1,   -- characters 1‑3
            substr(id_str, 4, 3) AS part2    -- characters 4‑6
        FROM padded
    )
    SELECT
        format(
            'price-tags/%s/%s/%s.webp',
            part1,
            part2,
            id_str
        )
    FROM parts;
$$;
```

Then, a SQL query was constructed to collect a diverse set of price tag images along with their associated data for training the visual language model.
The price tag distribution is as follows:

- 25,000 samples from France, all associated with a price
- all available samples associated with a price from other countries
- 4,000 samples with invalid price tags (unreadable, truncated, not a price tag, no barcode,...). These allow the future trained model to correctly handle such cases.

We first fetch all price tag IDs that are part of the benchmark to exclude them from the training dataset. In the `llm-evals` repository:

```bash
yq e -o=json llm_evals/tasks/prices/price_tag_extraction/dataset.yaml | jq -r '.cases[] | .name' | tr '\n' ',' | sed 's/,/),(/g' | sed 's/^/(/' | head -c -2```
```

In a nutshell:

- `yq` converts the YAML dataset file to JSON
- `jq` extracts the `name` field from each case (which corresponds to the price tag ID)
- `tr` replaces newlines with commas
- `sed` formats the output to be suitable for SQL `VALUES` clause. We first replace every comma with `),(`, then add an opening parenthesis at the start (with `sed 's/^/(/'`) and remove the trailing `,(` at the end, by stripping the last 2 bytes.

This gives us a list of price tag IDs to exclude:

```bash
(127260),(122445),(124798),(124154),(121551),(126568),(126807),(126492),(127190),...(76716)
```

Finally, the training dataset is collected with the following SQL query, excluding the benchmark price tags:

```sql
COPY
  (
    SELECT
      row_to_json(t) AS result_json
    FROM
      (
        WITH
          price_tags_joined AS (
            SELECT
              price_tags.id as price_tag_id,
              price_tags.status as price_tag_status,
              'https://prices.openfoodfacts.org/img/' || get_price_tag_image_path (price_tags.id::int) AS image_url,
              prices.price,
              prices.price_per,
              prices.id as price_id,
              prices.product_code,
              prices.category_tag,
              prices.price_without_discount,
              prices.discount_type,
              prices.origins_tags,
              prices.labels_tags,
              locations.osm_address_country_code AS country_code
            FROM
              price_tags
              JOIN proofs ON price_tags.proof_id = proofs.id
              JOIN locations ON proofs.location_id = locations.id
              LEFT JOIN prices ON price_tags.price_id = prices.id
            ORDER BY
              random()
          ),
          excluded_ids AS (
            SELECT
              *
            FROM
              (
                VALUES
                  (PRICE_TAG_ID_1),
                  (PRICE_TAG_ID_2),
                  (PRICE_TAG_ID_3),
                  ...
              ) AS v (id)
          ) (
            SELECT
              *
            FROM
              price_tags_joined
            WHERE
              country_code = 'FR'
              AND price_id IS NOT NULL
              AND price_tag_id NOT IN (
                SELECT
                  *
                FROM
                  excluded_ids
              )
            LIMIT
              25000
          )
        UNION ALL
        (
          SELECT
            *
          FROM
            price_tags_joined
          WHERE
            country_code <> 'FR'
            AND price_id IS NOT NULL
            AND price_tag_id NOT IN (
              SELECT
                *
              FROM
                excluded_ids
            )
        )
        UNION ALL
        (
          SELECT
            *
          FROM
            price_tags_joined
          WHERE
            price_id IS NULL
            AND price_tag_status <> 1
            AND price_tag_id NOT IN (
              SELECT
                *
              FROM
                excluded_ids
            )
          LIMIT
            4000
        )
      ) t
  ) TO '/tmp/training_dataset.jsonl';
```

We save this output as JSON in a file named `training_dataset.jsonl`. Then, we transform it to the required format to submit a Gemini batch inference job:

```bash
jq -c '{key: .price_tag_id | tostring, parts: [{type: "image", data: .image_url}], meta: {country_code: .country_code, price_tag_status: .price_tag_status, price: .price, price_per: .price_per, product_code: .product_code, category_tag: .category_tag, price_without_discount: .price_without_discount, discount_type: .discount_type, origins_tags: .origins_tags, labels_tags: .labels_tags}}' training_dataset.jsonl > training_dataset_formatted.jsonl
```

We save the schema and instructions in `price_tags_extraction/__init__.py` with the following content:

```python
import enum
from typing import Literal

from pydantic import BaseModel, Field


class Unit(enum.StrEnum):
    KILOGRAM = "KILOGRAM"
    LITER = "LITER"
    UNIT = "UNIT"


class DiscountType(enum.StrEnum):
    QUANTITY = "QUANTITY"  # example: buy 1 get 1 free
    SALE = "SALE"  # example: 50% off
    SEASONAL = "SEASONAL"  # example: Christmas sale
    LOYALTY_PROGRAM = "LOYALTY_PROGRAM"  # example: 10% off for members
    EXPIRES_SOON = "EXPIRES_SOON"  # example: 30% off expiring soon
    PICK_IT_YOURSELF = "PICK_IT_YOURSELF"  # example: 5% off for pick-up
    SECOND_HAND = "SECOND_HAND"  # example: second hand books or clothes
    OTHER = "OTHER"
    NO_DISCOUNT = "NO_DISCOUNT"  # no discount applied


class LabelPrice(BaseModel):
    """One of the prices displayed on a price tag.

    For raw products (without barcode), if the price is indicated per weight
    but is not per kilogram (example: per 100g or per 500g), the price per
    kilogram should be calculated.
    """

    price: float = Field(description="Price in the local currency")
    with_vat: bool = Field(
        description="True if the price includes VAT (Value Added Tax), false otherwise. "
        "If there is no VAT in the country, set to false. In most cases, this should be set to true.",
    )
    currency: str | None = Field(
        description="Currency of the price. Set to null if unknown. Examples: 'EUR', 'USD', 'GBP'",
    )
    price_per: Unit = Field(
        description="Unit of the price. Can be one of: KILOGRAM: price is per kg, only if type = CATEGORY; "
        "LITER: price is per liter, only if type = CATEGORY; "
        "UNIT: price is per unit (available for both CATEGORY and PRODUCT types)",
    )
    price_is_discounted: bool = Field(
        description="true if this particular price entry is a discounted price, false otherwise",
    )
    discount_type: DiscountType = Field(
        description="The type of discount applied to the price, if any. "
        "If no discount is applied, this should be set to NO_DISCOUNT. "
        "Possible discount types are: "
        " - QUANTITY: example: buy 1 get 1 free, "
        " - SALE: example: 50% off, "
        " - SEASONAL: example: Christmas sale, "
        " - LOYALTY_PROGRAM: example: 10% off for members, "
        " - EXPIRES_SOON: example: 30% off expiring soon, "
        " - PICK_IT_YOURSELF: example: 5% off for pick-up, "
        " - SECOND_HAND: example: second hand books or clothes, "
        " - OTHER: other types of discounts.",
    )
    uncertain: bool = Field(
        description="true if the price is uncertain: "
        "1) if the price is occluded (even partially), or blurred, "
        "2) if there is a reflection preventing accurate price reading, "
        "3) if the image quality is not good enough to read the price. "
        "Otherwise, the value must be false.",
    )


class Label(BaseModel):
    """A label (also called price tag) indicates in a store the price of the
    product and possibly other information such as the category, price per kg,
    origins, etc.

    We distinguish between two types of labels:

    - Labels for packaged products, with barcode (type: PRODUCT): these are
    products that have a barcode, which is usually displayed on the price tag.
    For this type of label, the category and origins should be set to null.
    - Raw products, without barcode (type: CATEGORY): these are usually fruits
    and vegetables, but it can also be any product sold per weight (kg, 100g,
    etc.). For this type of label, the category should be set to the
    corresponding category, the origins should be set to the countries of
    origin of the product (if indicated) and the barcode should be null.
    """

    type: Literal["PRODUCT", "CATEGORY"] | None = Field(
        description="The type of product the label is referring to. It should be "
        "`PRODUCT` for packaged products with barcode, and `CATEGORY` for raw "
        "products without barcode. If the type is unknown, this should be set to null.",
    )
    category: str | None = Field(
        description="The category of the product. If type=CATEGORY, this should be set to the "
        "category of the product, such as Apples, Bananas, Tomatoes, etc. The category must be in English, "
        "even if the category is displayed in another language on the price tag. "
        "The category must be the most precise category possible: for example, if the label says 'Red Onions', "
        "category should be 'Red onions', and not 'Onions'. "
        "If more than one category is present on the price tag, only the first category should be considered, and "
        "`has_multiple_categories` should be set to true. "
        "If unknown, or if TYPE=PRODUCT, this should be set to null.",
    )
    has_multiple_categories: bool | None = Field(
        description="If type is CATEGORY: if more than one category is present on the price tag, this should be true, "
        "false otherwise. If type is PRODUCT: it should be null",
    )
    prices: list[LabelPrice] = Field(
        description="All prices found on the label. Depending on the type of "
        "price tag, there can be multiple prices displayed. "
        "For packaged products (type=PRODUCT), the price per unit is the most common one. "
        "The price per kg is also often included. "
        "For raw products (type=CATEGORY), the price per kg is the most common one. "
        "There can also be a discount applied to the price. In such a case, both "
        "the original price and the discounted price should be included in the list.",
    )
    origins: list[str] | None = Field(
        description="The countries of origin of the product, in English. "
        "If type=PRODUCT, this should be set to null. If type=CATEGORY, this should "
        """be set to the countries of origin of the product, such as ["France"], ["Italy"], """
        """["Spain"], etc. """
        "Most of the time, there is only one country of origin indicated on the label, "
        "but sometimes there can be multiple countries (for example: 'France and Spain'), in "
        "which case all countries should be included in the list. "
        "If type=CATEGORY and the origin is unknown, it should be set to null.",
    )
    organic: bool | None = Field(
        description="true if the product is organic, false otherwise. If true, "
        "there should be evidence on the label suggesting that the product is "
        "organic, such as the EU organic logo or other recognized organic certifications. If the organic status is unknown, "
        "it should be set to null.",
    )
    barcode: str | None = Field(
        description="The barcode of the product, if available. "
        "The barcode is usually a number with 13 (EAN13) or 8 (EAN8) digits. You should "
        "*NOT* try to decode the barcode stripe (also called modules), but use the "
        "barcode number displayed on the label. "
        "If type=CATEGORY, this should be null.",
    )
    product_name: str = Field(
        description="The name of the product, as displayed on the label. "
        "For raw products (type=CATEGORY), this is usually the name of the fruit or vegetable. "
        "For products with barcode (type=PRODUCT), it usually includes the brand, a short "
        "description of the product, and optionally the quantity. "
        "If no product name is displayed on the label, this should be an empty string. "
        "examples: 'NOCCIOLATA BIO 650G', 'Simpson Donuts', 'GERBLE BISCUIT PIST ABRICOT160G', "
        "'Radis Blanc', 'Concombre lisse', 'Courget Butternut', 'Tomatoes', 'Organic Bananas'",
    )
    uncertain_barcode_or_product_name: bool = Field(
        description="true if the barcode (for type=PRODUCT) or category (for TYPE=CATEGORY) is uncertain: "
        "1) if the barcode (respectively the product name) is occluded (even partially), or blurred,"
        "2) if there is a reflection preventing accurate reading, "
        "3) if the image quality is not good enough to read the barcode (respectively the product name). "
        "Otherwise, the value must be false.",
    )
    is_price_tag: bool = Field(
        description="indicates whether the image shows a physical price tag attached to a product. "
        "For example, if the image seems to come from a receipt or a catalogue "
        "(or is a random image), this should be set to false.",
    )


OUTPUT_SCHEMA = Label

INSTRUCTIONS = "Here is one picture containing a price label, extract the required information from it. The output must be in JSON format conforming to the provided schema."
```

Then, we upload images to Google Cloud Storage, and obtain a new dataset file with URIs pointing to the GCS locations:

```bash
PYTHONPATH=`pwd` labelr google-batch generate-dataset --data-path training_dataset_formatted.jsonl --output-path training_dataset_google_batch.jsonl --config-module price_tags_extraction --thinking-level MINIMAL
```

Then, we launch the batch job:

```bash
labelr google-batch launch-batch-job price-tag-extraction-training-dataset --dataset-path training_dataset_google_batch.jsonl --model gemini-3-flash-preview --location global
```

We shuffle the dataset:

```bash
sort --parallel=8 -R gemini-batch_price-tag-extraction-training-dataset_prediction-model-2025-12-30T16_56_33.102348Z_predictions.jsonl > gemini-batch_price-tag-extraction-training-dataset_prediction-model-2025-12-30T16_56_33.102348Z_predictions_shuffled.jsonl
```

Upload the dataset (train split only):

```bash
GOOGLE_CLOUD_PROJECT=robotoff labelr google-batch upload-training-dataset-from-predictions gemini-batch_price-tag-extraction-training-dataset_prediction-model-2025-12-30T16_56_33.102348Z_predictions_shuffled.jsonl --instructions-path instructions.txt --json-schema-path schema.json --repo-id openfoodfacts/price-tag-extraction --tmp-dir /mnt/nvme/price_tag_ds_tmp --image-max-size 1024
```

We also need to upload the validation set, which corresponds to the benchmark present in llm-evals. We first convert the dataset YAML file in a format usable by labelr:

```bash
yq e -o=json dataset.yaml | jq -c '.cases[] | {image_id: .name, image_url: .inputs.image_urls[0]}' > dataset_val.jsonl
```

Here, we created a JSONL file with `image_id` and `image_url` fields.

Then, we upload the validation set to Hugging Face:

```bash
uv run labelr datasets export-llm-ds --dataset-path dataset_val.jsonl --repo-id openfoodfacts/price-tag-extraction --split val --image-max-size 1024
```

## Training

### 2026-01-13

A first training run was launched with the following command:

```bash
uv run main.py train --ds-repo-id openfoodfacts/price-tag-extraction --output-repo-id openfoodfacts/price-tag-extractor --preprocess-num-proc 8 --preprocess-writer-batch-size 2000
```

`--preprocess-num-proc 8` and `--preprocess-writer-batch-size 2000` allow to make the dataset preprocessing much faster. Adapt the values to how memory-hungry the preprocessing is.

The following environment variables were added to a .envrc file:

- `HF_TOKEN`: token for HF Hub, it's used to push the model weights
- `WANDB_API_KEY: API for Wandb, used for training tracking
- `WANDB_PROJECT`, `WANDB_NAME`: configure the Wandb project and run name respectively

### 2026-01-14

The training is over.

[wandb run](https://wandb.ai/raphaeloff/price-tag-extraction/runs/6dl5xoww) - [HF model](https://huggingface.co/openfoodfacts/price-tag-extractor)

We run the validation:

```bash
uv run main.py validate --ds-repo-id openfoodfacts/price-tag-extraction --lora-repo-id openfoodfacts/price-tag-extractor --output-path val.jsonl --base-model unsloth/Qwen3-VL-8B-Instruct --no-enforce-schema
```

There were two issues with the current version of the dataset:

- training samples were not shuffled, the sample order is the same as the order in which the dataset was generated: the samples are grouped by country. This may affect negatively performance. We can shuffle the dataset dynamically in the training script, but it takes time and requires unecessary disk space/network bandwidth (and with GPU rental, time/storage/network is money).
- images were kept as it and not resized. We resized them in the training script, which takes time.

I uploaded a new version of the dataset, with shuffled samples and all images with maximum size of 1024 (height or width).

I also added the following improvements to the training script (effective during next run):

- add a default warmup ratio (ratio of number of steps) of 0.1, instead of a fixed `warmup_steps` value of 5
- by default, don't shuffle the dataset (we did this before uploading the dataset)
- set recommended default values recommended by Unsloth:
    - `weight_decay=0.01` (instead of 0.001)

LoRA dropout value was kept to 0.0, as Unsloth training is optimized for a LORA dropout of 0.0.

Benchmark results (v2.0) using `llm-evals`:

```
Detailed scores:
  price: 499/523 (95.41% accuracy)
  barcode: 360/408 (88.24% accuracy)
  uncertain_barcode_or_product_name: 488/528 (92.42% accuracy)
  category: 0/55 (0.00% accuracy)
```

For reference, the original model:
```

Detailed scores:
  price: 466/523 (89.10% accuracy)
  barcode: 345/408 (84.56% accuracy)
  uncertain_barcode_or_product_name: 497/528 (94.13% accuracy)
  category: 19/66 (28.79% accuracy)
```

So we get:
- price: +4.3% (89.10 > 95.41%)
- barcode: +3.6% (84.56 > 88.24%)
- uncertain_barcode_or_product_name: -1.7% (94.13 > 92.42%)
- category: -28.79% (28.79 > 0%)

The 0% accuracy for category arises from the fact the category names predicted by the fine-tuned model are in their original language (ex: French, German,...). It is the case as well for all samples of type `CATEGORY` in the training set, which explains this behavior.

After further analysis, it turns out the `origins` field are always in their original language as well.


## 2025-01-19 - 2025-01-23

I fixed the translation issues in the `category` and `origins` fields by:

- automatically translate the category using gpt-oss-120b. After this automatic translation, a few issues were spotted and fixed manually.
- map ~80% of origins to English using a harcoded mapping, and translate the remaining using gpt-oss-120b.

I used [directus](https://directus.io/) (a headless CMS) locally to act as a backend as a service, in order to store and update easily the JSON data thanks to the provided API. The search functionality (using filters on any field) was also really convenient to spot issues and correct them using the UI.

I first created a `price_tag` collection and configured all the fields using the UI. I allowed public access (without authentication) by providing all permissions on the `price_tag` collection in the `Public` policy. I then generated a JSONL file from the dataset on the HF Hub, and uploaded the data on directus:

```bash
uv run labelr directus upload-data --dataset-path dataset.jsonl --collection-name price_tag
```

I exported the data from directus using labelr:

```bash
uv run labelr directus export-data --output-path - --collection price_tag | jq -c '{image_id: .price_tag_id, output: {type, prices, origins, organic, barcode, product_name, category, has_multiple_categories, is_price_tag, uncertain_barcode_or_product_name}}' > updated_train_dataset.jsonl
```

I then uploaded the new dataset version to the HF Hub:

```bash
uv run labelr datasets update-llm-ds --dataset-path updated_train_dataset.jsonl --repo-id openfoodfacts/price-tag-extraction --split train
```

I used the `--show-diff` to verify the updates before uploading the new version of the dataset (before actually uploading it).

A new training run was launched with higher LORA k value (`k=32`), and with the updated default values for hyperparameters (warmup ratio of 0.1, weight_decay of 0.01, shuffled dataset):

```bash
uv run main.py train --ds-repo-id openfoodfacts/price-tag-extraction --output-repo-id openfoodfacts/price-tag-extractor --lora-r 32 --lora-alpha 32 --preprocess-num-proc 8 --preprocess-writer-batch-size 2000
```