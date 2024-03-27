# Spellcheck


## Benchmark - Validation dataset

To improve the quality of the Spellcheck module, we decided to exploit the recent advancements with LLMs to train a task-specific Machine Learning model on OFF data. 

Creating this kind of solution requires to rigoursly build a benchmark/validation dataset to estimate the future models performances. 

Our idea is to use the existing dataset developed a few years ago, enhancing it with new data, then perform data-augmentation using LLMs.

Not only do we build a benchmark to evaluate future solutions, we'll use data augmentation to create the dataset required to train a task-specific machine learning model.

### Data lineage

Old dataset, located at `spellcheck/old/test_sets/fr` is leveraged to constitute our new dataset. It is composed of `List of Ingredients` before and after spellcheck, mainly in French.

We take this dataset and process it. 
The processing scripts are located at: `spellcheck/scripts/old_to_new` - and the processed data: `spellcheck/data`.

* `0_fr_data.json`: the old data is extracted and transformed into a json file.  Basic processing are performed, such as removing *NO_VALID* data (data size: **786**) - script: `0_convert_old_data.py`

* `1_old_fr_no_duplicate_data.json`: We noticed a lot of duplicates before and after spellcheck, representing almost half of the dataset. We remove them (data size: **441**) - script: `1_old_fr_data_check.ipynb`