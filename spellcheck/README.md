# Spellcheck

## ⭐ Guidelines

The influence of the Spellcheck on the list of ingredients needs to be controlled to avoid alterating contributions and/or add new errors. Therefore, we keep the modification to a minimum to favour Precision over Recall.

From the different types of errors observed across products, we came up with these spellcheck guidelines:

* Correct typos;
* Percentages
    * Whitespaces between words and percentages shouldn't be corrected. The text needs to be kept as unchanged as possible.
    (Example: `Ingredient 0,2   %`).
    * The only case when a whitespace involving a percentage should be modified is if the *digit* is stuck in the previous word (*ex: cheese1.9% -> cheese 1.9%*)
* Some ingredients are enclosed with `_`, such as `_milk_` or `_cacahuetes_`, to detect allergens. Should remain unchanged. However, in the case it is not an ingredient, such as `_Cacahuetes_ con cáscara tostado. _Trazas de frutos de cáscara_.`, it needs to be modified into `_Cacahuetes_ con cáscara tostado. Trazas de frutos de cáscara.`;
* Some percentages were badly parsed by the OCR, such as `cheese (196)` instead of `cheese (1%)` or `καραμέλα (396` instead of `καραμέλα (3%)` . Since there is a recognizable pattern, `%` being transformed into `96`, the spellcheck should be able to recognize and correct it.
* If characters in French miss an accent, needs to be fixed. (*ex: cafe -> café*)
* `*` should remain in the corrected text as much as possible (*ex: Schweinefleisch\* -> Schweinefleisch\**)
* Whitespaces shouldn't been modified except for this cases:
    * Words or characters that are supposed to be separated: *ex: crabe(...) -> crabe ()*;
    * No whitespace after a punctuation (*ex: syrup,sugar -> syrup, sugar*)
* Uppercase to lowercase or vice-versa are accepted.
* In French, the character `oe` or `œ` should remain unchanged after correction (*ex: œuf, bœuf). If it is missing, should be replaced by default by `œ`.

## ✅ Benchmark - Validation dataset

To improve the quality of the Spellcheck module, we decided to exploit the recent advancements with LLMs to train a task-specific Machine Learning model on OFF data. 

Creating this kind of solution requires rigorously building a benchmark/validation dataset to estimate the future models' performances. 

Our idea is to use the existing dataset developed a few years ago, enhance it with new data, and then perform synthetic data generation using LLMs.

### 🧵 Data lineage

*Data*
```bash
├── data
│   ├── benchmark
│   │   ├── benchmark.json
│   │   ├── test_benchmark.json
│   │   └── verified_benchmark.parquet
│   ├── fr
│   │   ├── 0_fr_data.json
│   │   └── 1_old_fr_no_duplicate_data.json
│   └── labeled
│       └── corrected_list_of_ingredients.txt
```

*Scripts*
```bash
├── scripts
│   ├── argilla
│   │   ├── benchmark.py
│   │   └── extract_benchmark.py
│   ├── benchmark
│   │   ├── create_benchmark.py
│   │   ├── create_test_benchmark.py
│   │   └── evaluation.py
│   └── old_to_new
│       ├── 0_convert_old_data.py
│       └── 1_old_fr_data_check.ipynb
```

The benchmark is composed of **243** lists of ingredients from 3 data sources:

* **30%** of the old dataset composed of manually corrected lists of ingredients in French from the previous work by Lucain W. The old dataset, `spellcheck/old/test_sets/fr` is used  to constitute our new dataset. It is composed of `List of Ingredients` before and after spellcheck, mainly in French. The processing scripts are located at: `spellcheck/scripts/old_to_new` - and the processed data in `spellcheck/data/fr`.

    * `data/fr/0_fr_data.json`: the old data is extracted and transformed into a json file. Basic processing are performed, such as removing *NO_VALID* data (data size: **786**) - script: `scripts/old_to_new/0_convert_old_data.py`

    * `data/1_old_fr_no_duplicate_data.json`: We noticed a lot of duplicates before and after spellcheck, representing almost half of the dataset. We remove them (data size: **441**) - script: `scripts/old_to_new/1_old_fr_data_check.ipynb`

* **15** manually corrected lists of ingredients in different languages. 
    
    * This small sample was used to prompt engineer GPT-3.5 on achieving good performances on the spellcheck task.
    * Those examples mainly comes from the OFF website.
    * data: `data/labeled/corrected_list_of_ingredients.txt` & `data/benchmark/test_benchmark.json` - script: `scripts/benchmark/create_test_benchmark.py` 

* **100** lists of ingredients with the tag `50-percent-unknown` corrected with the prompted GPT-3.5. It follows the correction guidelines defined with the OFF team and based on observations in production. You'll find the prompt used to augment the data with GPT-3.5 at `utils/prompt.py`. These 100 lists of ingredients are extracted from the OFF database and processed right away during the benchmark creation.

Benchmark composition script: `scripts/benchmark/create_benchmark.py`

Once composed, the benchmark is then verified  using Argilla to ensure the correction generated by OpenAI respect the Spellcheck guidelines. The corrected benchmark is located at `data/benchmark/verified_benchmark.parquet`.

### ✍️ Argilla

Argilla is an open-source annotation tool specific to Natural Language Processing.

To annotate and verify the benchmark, we deployed an Argilla instance and manually verified the correction generated by GPT-3.5.

Scripts:
* `scripts/argilla/benchmark.py`: structure of the annotation tool for the spellcheck task
* `scripts/argilla/extract_benchmark.py`: script to extract the annotated dataset from Argilla. The extracted dataset is saved at `data/benchmark/verified_benchmark.parquet`.


### 📐 Evaluation metrics and algorithm

Evaluating the Spellcheck is a hard task. 

Most of the existing metrics and evaluation algorithms compute the similarity between the reference and the prediction such as BLEU or ROUGE scores. Others calculate the Precision-Recall on modified tokens for token classification tasks.

But in our case, we would like to estimate how well the Spellcheck performs on recognizing and fixing the right elements in the list of ingredients. Therefore we need to compute the Precision-Recall of correctly modified tokens. 

However, we don't have access to these tokens. Only to these text sequences:
* The original: the list of ingredients to be corrected;
* The reference: how we expect this list to be corrected;
* The prediction: the correction from the model.

Is there any way to get the Precision-Recall scores on corrected tokens only from these sequences? The answer is yes. This is the function of our evaluation algorithm: `scripts/benchmark/evaluation.py`.

The uniqueness of this evaluation algorithm lies in its calculation of precision, recall, and F1 scores specifically for errors to be corrected, which are directly extracted from texts.

The process is divided into 4 steps:

1. **Texts (Original-Reference-Prediction) are tokenized using a Byte Pair Encoding ([BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding)) tokenizer from the [tiktoken](https://github.com/openai/tiktoken) library from OpenAI.**

Example:
``` 
Original:       "Th cat si on the fride,"
Reference:      "The cat is on the fridge."
Prediction:     "Th big cat is in the fridge."
```

After tokenization:
``` 
Original:       1016   8415   4502   389   279   282     1425   11
Reference:      791    8415   374    389   279   38681   13
Prediction:     1016   2466   8415   374   304   279     38681  13
```

We notice which tokens were modified, added, or deleted. But this transformation creates a misalignement. Thus, we need to align those 3 token sequences. 

2. **Encoded originals and references are aligned using a [Sequence Alignment](https://en.wikipedia.org/wiki/Sequence_alignment) technique. This kind of algorithm is particularly used in bioinformatics to align DNA sequences.**

We use the [Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm) algorithm and create 2 lists of pairs: *Orig-Ref pairs* and *Orig-Pred pairs*. 

After alignment:
``` 
Original:       1016   8415   4502   389   279   282    1425   11
Reference:      791    8415   374    389   279   38681   --    13

Original:       1016    --    8415   4502  389   279    282    1425   11
Prediction:     791    2466   8415   374   304   279    --     38681  13
```

Now we can detect which tokens were added, deleted or modified from the *Original* sequence for both the *Reference* and the *Prediction*. 

But as you may have noticed, pairs of tokens are now misaligned because a new word `big` (`2466`) was added to the Prediction but not in the Reference.
 
3. **Pairs of tokens (Original-Reference; Original-Prediction) are aligned to consider gaps in case Reference and/or Prediction have different lengths.**

This mainly occurs when additional words are added whether in References or Predictions compared to Originals. This is translated as an additional *gap* in the Original list of tokens.

To better visualize which tokens were modified in comparison of the *Original*, each list of pairs is modified into a sparse vector. If the original token was modified, or if a token was added or deleted, it is considered as `1`: 

``` 
Orig-Ref:      1   0   1   0   1   1   1   1
Orig-Pred:     0   1   0   1   1   1   1   1   1
```

Since the token `2466` was added in *Orig-Pred pairs*, we insert `0` into the *Orig-Ref sparse vector*, the shortest vector in this case, meaning that this "imaginary" token does not count as a change.

``` 
Orig-Ref:      1  '0'  0   1   0   1   1   1   1
Orig-Pred:     0   1   0   1   1   1   1   1   1
```

--------------------------------------------------------------------
*Note:* We would do the same in case an additional token were added to the Reference instead, or in both Reference and Predictions. Here's an example of the latter: 

*Before*
``` 
Original:       1016   8415   4502   389    279    --     282    1425    11
Reference:      791    8415   374    389    279    9999   38681   --     13
Sparse:         1      0      1      0      0      1      1      1       1       

Original:       1016    --    8415   4502   389    279    282    1425    11
Prediction:     791    2466   8415   374    304    279    --     38681   13
Sparse:         1      1      0      1      1      0      1      1       1
```

*After*
``` 
Original:       1016    --    8415   4502   389   279     --    282    1425    11
Reference:      791     --    8415   374    389   279    9999   38681   --     13
Sparse:         1       0     0      1      0      0     1      1      1       1  

Original:       1016    --    8415   4502   389   279    --     282    1425    11
Prediction:     791    2466   8415   374    304   279    --     --     38681   13
Sparse:         1      1      0      1      1      0      0      1     1       1
```
--------------------------------------------------------------------

Our pairs are now aligned. We can now know which tokens were supposed to change, and which were not supposed to.

By multiplying the sparse vectors, we can calculate the Precision-Recall metrics.

4. **Compute Precision, Recall, and Correction Precision**

By taking these 2 sparse vectors and their inverse, we can calculate the number of True Positives (`TP`), False Positives (`FP`) and False Negatives (`FN`) to compute the Precision and Recall.

If we consider the sparse vector corresponding to the *Prediction*:

```
Orig-Ref:          1    0    0    1    0    1    1    1    1
Orig-Pred:         0    1    0    1    1    1    1    1    1
Signification:     FN   FP   TN   TP   FP   TP   TP   TP   TP
```

Also, since these metrics consider if the correct token was modified, and not if the right token was chosen by the 
model, we also calculate the `correction_precision` for each `TP`.

`correction_precision` considers only tokens modified by the model **that were supposed to be modified according to the Reference.**

With these metric, we're now capable of evaluating our spellcheck accurately on this task!

**Notes:**
* This evaluation algorithm depends on how well the sequence alignment was performed. It works only if there's enough information (similar tokens) to align sequences. It means noisy sequences can influence the sequence alignment and therefore bias the metrics calculation. Adding a noise threshold, such as calculating the **BLEU** score between Original-Reference & Original-Prediction could be a good solution to prevent this.

* The Needleman-Wunsch is the foundation of this algorithm. It can be worth performing hyperparameter tuning to get the best sequence alignment for our case.


### 👨‍⚖️ LLM evaluation against the benchmark

We evaluated **Proprietary LLMs** such as OpenAI GPTs and Anthropic Claude 3 models. This gives us a baseline on how these solutions perform on the Spellcheck task compared to our model.

Benchmark version: **v0.2**


| Model | Precision | Recall | F1-Score | Correction Precision|
|----------|----------|----------|----------|----------|
| GPT-3.5-Turbo | **0.487** | **0.567** | **0.501** | **0.541** | 
| GPT-4-Turbo | 0.401 | 0.535 | 0.430 | 0.490 |
| Claude-3-Haiku | 0.322 | 0.465 | 0.345 | 0.405 | 
| Claude-3-Sonnet | 0.366 | 0.490 | 0.383 | 0.439 |
| Claude-3-Opus| 0.449 | 0.481 | 0.435 | 0.516 |

Notes:
* **Precision**: Proportion of model predictions that were actually mistakes
* **Recall**: Proportion of mistakes founded
* **F1-Score**: Mean-like between Precision and Recall
* **Correction Precision**: When model found the mistake location, proportion of right modification 

## Training dataset 

### Extract the data

From the JSONL file available on [Open Food Facts Data](https://world.openfoodfacts.org/data), we extracted **3000** products. We decided to select a percentage-unknonwn-ingredients between *20% - 40%*.

Since this tag doesn't exist, we calculated the percentage-unknown using the keys `fraction` = `unknown-ingredients_n` / `ingredients_n`. 

The dataset being extremely large (43 GB once decompressed), we used the [Polars](https://pola.rs/) library to manipulate the data. You can find the extraction script at `scripts/dataset/extract_data.py`.

The extracted products are stored as a `.parquet` file at `data/dataset/extracted_lists_of_ingredients.parquet`.

### Generate the synthetic data

We then generated the synthetic dataset using GPT-3.5-Turbo and the same prompt that was used for generating the benchmark, located at `utils/prompt.py`.

Calling OpenAI GPT-3.5-Turbo to constitute our dataset costed around **3.25$** (around 6 millions tokens - ~3000 requests).

The script is located at `scripts/dataset/generate_synthetic_data.py` and the synthetic dataset at `data/dataset/synthetic_data.jsonl`.

### Argilla

To get an overview of the dataset and later correct it manually, we pushed the synthetic dataset into **Argilla**.

You can find it [there](https://argilla.openfoodfacts.org/dataset/6ecc7c73-900b-4557-a12d-4ab23266a681/annotation-mode).

### Post-processing

After a first check on Argilla, there are plenty of *low-hanging fruits* errors implemented during the synthetic data generation that we can correct using a post-processing step:

* `###Corrected list of ingredients:` from the prompt in the output

The post-processed data is located at `data/dataset/post_processed_synthetic_data.jsonl` (*manually processed*)
