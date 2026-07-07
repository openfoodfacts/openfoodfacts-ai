# Price tag classification

Once we detected price tags using the object detection model, we would like to auto-categorize them using an image classification model to speed up manual processing, and let contributors focus on high-value tasks.

Following [this discussion](https://github.com/openfoodfacts/open-prices/discussions/1234), we will classify each price tag based on its perceived "quality":
- `invalid`: high blurriness, or the object is not a price tag, or at least half the price tag is truncated.
- `medium-quality`: low to medium blurriness (the price and barcode can possibly be read), or the price tag is at least partially truncated (but more than 1/2 of the price tag area is visible).
- `high-quality`: very low blurriness

## Classifier boundaries

The model classifies price tags based:
	- global blurriness of the image
	- whether the price tag is truncated or not
	- whether the image is a price tag or not

We should not expect the classifier to be able to perform OCR or to validate the price tag content.
For example, [this image](https://prices.openfoodfacts.org/img/price-tags/000/137/000137193.webp) is classified as `high-quality`, even the price tag content is empty, as the image quality is good and as the price tag is not truncated.
Similarly, images that look like possible price tags but are not are classified only based on the image blurriness and possible truncation. For example, [this image](https://prices.openfoodfacts.org/img/price-tags/000/181/000181909.webp) is classified as `high-quality`, as:
	- without knowing what's written on the leaflet, it could possibly be a price tag
	- the image quality is good
	- it's not truncated

The rational behind this is to prevent the classifier to discard spuriously price tags: in doubt, let's run the price tag extraction model to see if we can extract something from the image.
## Data preparation

### `invalid` and `medium-quality` labels

I start by collected price tags belonging to the `invalid` class, by selecting price tags with the following characteristics:

- `status=2` (unreadable) or `status=4` (not a price tag)
- without the `prediction-barcode-valid` tag: if the extraction model detected a barcode that is valid, it's probable the contributor classified it with `status={2,4}` by mistake
- only keep price tags from the latest model (`gemini-3-flash-preview`), with `type = PRICE_TAG_EXTRACTION`, and with the latest schema (`2.0`).

The SQL query is the following:

```sql
SELECT
  t1.id,
  'https://prices.openfoodfacts.org/img/' || public.get_price_tag_image_path(t1.id::integer) as image_url,
  t1.created,
  t1.status,
  t1.updated_by,
  t1.price_id,
  t1.proof_id,
  t1.tags,
  (t2.data #> '{selected_price,price}') IS NOT NULL as has_predicted_price,
  (t2.data ->> 'barcode'::text) <> '' as has_predicted_barcode,
  (t2.data ->> 'category') IS NOT NULL as has_predicted_category,
  t2.data ->> 'truncated' as predicted_truncated,
  (t2.data ->> 'blurriness')::float as predicted_blurriness,
  t4.osm_address_country_code
FROM
  price_tags as t1
  JOIN price_tag_predictions as t2 ON t2.price_tag_id = t1.id
  JOIN proofs as t3 on t1.proof_id = t3.id
  JOIN locations as t4 on t3.location_id = t4.id
WHERE
  t1.status IN (2, 4)
  AND NOT 'prediction-barcode-valid' = ANY (t1.tags)
  AND t2.model_version = 'gemini-3-flash-preview'
  AND t2.type = 'PRICE_TAG_EXTRACTION'
  AND t2.schema_version = '2.0';
```

For reference, the `get_price_tag_image_path`function is defined as:

```sql
CREATE
OR REPLACE FUNCTION public.get_price_tag_image_path (p_price_tag_id integer) RETURNS text LANGUAGE sql AS $function$
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
$function$
```

We extract useful information detected by Gemini:
- `has_predicted_price` `has_predicted_barcode`, `has_predicted_category`: if we have a selected price, barcode or category extracted, respectively
- `predicted_blurriness`: the blurriness score, between 0 (not blurry) and 1 (extremely blurry)
- `predicted_truncated`: whether the price tag is detected as truncated

These information will be used to add tags to each Label Studio sample and simplify annotation.
We also extract the country code, as it can be useful later to filter price tags of a specific country.

4486 price tags were extracted using this query, saved in `status_2-4.json`.
I convert the data to JSONL:

```bash
jq -c '.[]' status_2-4.json > status_2-4.jsonl
```

Then convert to a format accepted by Labelr:

```bash
cat status_2-4.jsonl | jq -c '{image_url, image_id: .id, tags: ["status:\(.status)", "has_predicted_price:\(.has_predicted_price)", "has_predicted_barcode:\(.has_predicted_barcode)", "has_predicted_category:\(.has_predicted_category)", "predicted_truncated:\(.predicted_truncated)", "predicted_blurriness:\(.predicted_blurriness)", "country_code:\(.osm_address_country_code)"], proof_id, price_id}' > to_import_status_2-4.jsonl
```

This format is compatible with what's expected by the `create-dataset-file` command. We add most useful information as tags, so that we can use these tags to filter in Label Studio interface.

I created a new project (`ID=66`) in Label Studio with the following config:
```xml
<View>
  <Image name="image" value="$image_url"/>
  <Choices name="choice" toName="image">
    <Choice value="invalid"/>
    <Choice value="low-quality" />
    <Choice value="medium-quality" />
    <Choice value="high-quality" />
  </Choices>
</View>
```

Then, I created the dataset file:

```bash
labelr ls create-dataset-file --input-file to_import_status_2-4.jsonl --output-file ls_to_import_status_2-4.jsonl
```
After performing a random sort (with `sort -R`) and selecting the first 1000 lines (let's see later if we need more), I imported the data:

```bash
labelr ls import-data --project-id 66 --dataset-path ls_to_import_status_2-4_head_1000.jsonl
```

I then labeled most price tags, using the tags to group price tags that should have similar labels.

### `high-quality` labels

I use a slightly modified version of the previous query to select price tags that should probably be labeled as `high-quality`:

```sql
SELECT
  t1.id,
  'https://prices.openfoodfacts.org/img/' || public.get_price_tag_image_path(t1.id::integer) as image_url,
  t1.created,
  t1.status,
  t1.updated_by,
  t1.price_id,
  t1.proof_id,
  t1.tags,
  (t2.data #> '{selected_price,price}') IS NOT NULL as has_predicted_price,
  (t2.data ->> 'barcode'::text) <> '' as has_predicted_barcode,
  (t2.data ->> 'category') IS NOT NULL as has_predicted_category,
  t2.data ->> 'truncated' as predicted_truncated,
  (t2.data ->> 'blurriness')::float as predicted_blurriness,
  t4.osm_address_country_code
FROM
  price_tags as t1
  JOIN price_tag_predictions as t2 ON t2.price_tag_id = t1.id
  JOIN proofs as t3 on t1.proof_id = t3.id
  JOIN locations as t4 on t3.location_id = t4.id
WHERE
  t1.status = 1
  AND 'prediction-barcode-valid' = ANY (t1.tags)
  AND t2.model_version = 'gemini-3-flash-preview'
  AND t2.type = 'PRICE_TAG_EXTRACTION'
  AND t2.schema_version = '2.0';
```

The differences:
- `status = 1` to only select price tags that are linked to a price
- `'prediction-barcode-valid' = ANY (t1.tags)` to select price tags for which a valid barcode was extracted

41 339 items were extracted. I performed the same step as previously, the only difference being that I randomly sorted the file *before* calling `create-dataset-file` and picked the top 1500 lines, to speed the process.

After annotated all samples, I exported the dataset:
```bash
labelr datasets export --from ls --to hf \
--repo-id openfoodfacts/price-tag-classification \
--task-type classification \
--label-names invalid,medium-quality,high-quality \
--view-id 147 \
--meta-schema-path config_files/meta_schema_price_tag_image_cls.json \
--image-max-size 1024
```

with `config_files/meta_schema_price_tag_image_cls.json` content being:

```json
{
  "price_id": "string",
  "proof_id": "string"
}
```

View with ID 147 only contains annotated samples. 

I then run a [first training run](https://huggingface.co/openfoodfacts/price-tag-classification/tree/yolov8n-cls-e-100-custom-augmentation), with the following characteristics:
- 100 epochs
- image max size: 960px
- base model: yolov8n-cls
- custom classification

The model reached 94.8% accuracy on the validation set. Visual inspection of the predictions showed that model only failed on tricky samples. It was then integrated into Robotoff/Triton (https://github.com/openfoodfacts/robotoff/pull/1875) and Open Prices (https://github.com/openfoodfacts/open-prices/pull/1284).
