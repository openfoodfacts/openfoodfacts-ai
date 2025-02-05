# SpellCheck

The Open Food Facts SpellCheck project aims to correct misspelled ingredients on product descriptions. Those errors mainly appear because of the use of the OCR.  

- [Install dependencies](#install-dependencies)
- [Label data](#label-data)
- [Tests](#tests)
- [Models](#models)
- [Performances](#performances)

# Install dependencies

This project runs using Python3.
Create a virtualenv.
```
virtualenv -p /path/to/python3 .venv
```

Enter virtualenv.
```
source .venv/bin/activate
```

Install requirements.
```
pip install -r requirements.txt
```

### Vocabularies

The RegexModel uses vocabularies in order to make corrections (see **Models**). Vocabularies come from the [RobotOFF github repository](https://github.com/openfoodfacts/robotoff/tree/master/data/taxonomies). Please run download script to use them :
```
sh ./download_vocabulary.sh
```

# Label data

To label data, you must have a MongoDB running with the OFF database on it. By default, it will connect to localhost/port 27017 without any password. To change this behavior, edit the `mongo.py` file.  
Once MongoDB is configured, run the following command :
```
streamlit run label.py
```
It will open a streamlit dashboard on the browser.

# Tests

## Run tests

Command to run tests:
```
python test.py
```
For now, test is ran only on FR dataset for selected models. To configure which models to use, edit the `test.py` file.  

Each time the script is ran, a folder is created under `experiment/model_name/datetime/`. It contains :
- `original.txt` : the original descriptions sent to the model.
- `correct.txt` : the correct descriptions (hand-labelled).
- `prediction.txt` : the descriptions corrected by the model.
- `tags.txt` : tags associated to each item. Can be ignored.
- `metrics.json` : summary of the performances of the model.
- `detailed_metrics.csv` : CSV containing metrics for each individual item.

## Review predictions

A CLI tool is available to review model predictions once the test pipeline has been run. It takes the path to the experiment folder and an item_id as input. It outputs :
- the original, correct and predicted descriptions with normalization (lowercase, œu->oe,...)
- the original, correct and predicted ingredients Counter, as they are computed for the metrics
- some curated ingredients Counters with only relevant ingredients (those appearing in the calculation of precision/recall/fidelity metrics)

See example bellow :

```
>> python cli.py review --path=experiments/RobotoffAPI__index_product__conf_1/2020_04_22_17h17m15s --item-id=3321431025323

Id : 3321431025323

Original   : saumon (poisson) atlantique (salmo salar) élevé en ecoss (royaume-uni) issu d'animaux nourris sans ogm 97%, sel 3%.

Correct    : saumon (poisson) atlantique (salmo salar) élevé en ecosse (royaume-uni) issu d'animaux nourris sans ogm 97%, sel 3%.

Prediction : saumon (poisson) atlantique (salmo salar) élevé en ecosse (royaume-uni) issu d'animaux nourris sans ogm 97%, sel 3%.

Tags     : ['VALID']

Original ingredients :
Counter({'saumon': 1, 'poisson': 1, 'atlantique': 1, 'salmo': 1, 'salar': 1, 'élevé': 1, 'en': 1, 'eco': 1, 'royaume': 1, 'uni': 1, 'issu': 1, "d'": 1, 'animaux': 1, 'nourri': 1, 'san': 1, 'ogm': 1, '97': 1, 'sel': 1, '3': 1})

Correct ingredients :
Counter({'saumon': 1, 'poisson': 1, 'atlantique': 1, 'salmo': 1, 'salar': 1, 'élevé': 1, 'en': 1, 'ecosse': 1, 'royaume': 1, 'uni': 1, 'issu': 1, "d'": 1, 'animaux': 1, 'nourri': 1, 'san': 1, 'ogm': 1, '97': 1, 'sel': 1, '3': 1})
Predicted ingredients :

Counter({'saumon': 1, 'poisson': 1, 'atlantique': 1, 'salmo': 1, 'salar': 1, 'élevé': 1, 'en': 1, 'ecosse': 1, 'royaume': 1, 'uni': 1, 'issu': 1, "d'": 1, 'animaux': 1, 'nourri': 1, 'san': 1, 'ogm': 1, '97': 1, 'sel': 1, '3': 1})


Not original, correct
Counter({'ecosse': 1})

Not original, predicted

Counter({'ecosse': 1})
Not original, correct, predicted
Counter({'ecosse': 1})

Original, correct, not predicted
Counter()
```

# Models

The test framework is based on models. Each model is a class inherited from `BaseModel` that implements a Spellcheck algorithm. The Model can then be evaluated on the test data using the `test.py` script.

## Identity Model

Basic model that doesn't correct anything on the input data. The objective is to have a baseline against which to compare algorithms.

## RobotOFF API Model

Model that wraps calls to the [RobotOFF API](https://github.com/openfoodfacts/robotoff). RobotOFF corrections are mainly based on some ElasticSearch suggestions. Rules are then applied to determine whether or not the suggestion is relevant.  
The RobotOFF API has two parameters :
- `index` :  can be either `product` (default), `product_all`, `product_extended` or `product_extended_all`]. Specify the vocabulary used by ES.
- `confidence` : float, default to 1. Threshold parameter for ElasticSearch.


At the moment, RobotOFF API is the model that handles almost all the spelling corrections. Especially, it performs well on misspelled words (maximum edit distance : 2). Calls are cached using joblib to speed up the tests so make sure to refresh the cache if RobotOFF behavior changed.

## Regex Model

Model containing several rule-based algorithms.

### Pattern-based replacements

Most basic algorithm that does some search and replace in the original string.  
Patterns can be find in `./spellcheck/models/regex/patterns_fr.txt` . The file is divided in sections. Each section is composed of the correct form (first line) followed by common mistakes (following lines). The objective is to complete the document incrementally. Comments can be added using a `#` .  
**Example :**
```
# This is a comment

ingrédient
ingredienb
ingedients
ingédients
```

### Percentages

Regex-based rules that aim to handle all occurrences of percentages in a string and format them to a standard shape.

**Examples :**
```
0.1%   -> 0,1%
40, 3% -> 40,3%
15?.3% -> 15,3%
100 %  -> 100%
```

A dataset specific to percentages has been created (`./spellcheck/test_sets/percentages/fr.txt`). The goal is to have a way to be sure that the rules are doing what we want them to do and that adding a new one doesn't break the entire algorithm. This dataset is better suited for this task than the global metrics. A pytest script is associated to it. At the moment, the results are **1 failure against 1576 successes**. The remaining failure is an hard-example that we do not want to solve with a dedicated rule. If more mistakes of this type arise in the future, we might want to rethink it.

**Rules :**  
- First we match substrings following this pattern : `[0 to 2 digits][(optionally) a whitespace][a separator][0 to 2 digits][(optionally) a whitespace][a % or similar symbol]`.
- If the match contains/is part of an additive (e.g. Exxx), we drop it and to nothing.
- Depending on whether the first or last digits are present or not, we format the match accordingly (e.g `XX,XX%` or `XX%`).
- If no separator is found (only a whitespace), we concatenate the digits (example : `4 0%` -> `40%`). Just as a guarantee, we make sure that the value is below (or equals) to 100.
- In addition to these rules, we pad the match with whitespaces if context needs it. Pad is added if previous or next char is an alphanumerical character (`raisin7%` -> `raisin 7%`) or an opening/closing parenthesis (`19%(lait` -> `19% (lait`).

### Vocabulary corrections

This third method is based on dictionaries of known words. The first vocabulary, called WikipediaVoc, is extracted and curated from the Wikipedia dataset. This vocabulary is very large and contains a lot of rare words. Hopefully, it doesn't contain a lot of misspelled words since only words occurring more than 3 times are kept. Second vocabulary, called IngredientsVoc, is created from the OFF database. This voc is smaller but more specialized for foods. Vocabularies come from the [RobotOFF github repository](https://github.com/openfoodfacts/robotoff/tree/master/data/taxonomies). They need to be downloaded before running the model using the `download_vocabulary.sh` script (see **Install dependencies**).

Method used :
- We first tokenize the full description.
- For each alphabetical tokens, we check whether the token is a know word of WikipediaVoc.
- If not, 2 methods are used to make suggestions.

**Finding the right split**  
Sometimes the OCR misses a whitespace between two words (example : `farinede blé` -> `farine de blé`). In general, ElasticSearch is unable to suggest the good correction (example: `farine blé`). The idea is to look at every potential split (`fa rinede`, `far inede`, `fari nede`, (...), `farine de`) and keep it if both words exist in the IngredientsVoc (here : `farine de`). If multiple correct splits are possible, no changes is applied. Additionally, we don't consider splits that create a 1-char word in order to prevent introducing new errors.  
*NB:* This last rule is conservative and sometimes misses some corrections (example : `àcoque` -> `à coque`).

**Finding a correct variant**  
Sometimes the OCR outputs the good word but without accents. Example : `ingredients` -> `ingrédients`. For each unknown token, we check whether an accented variant of the token exists in the IngredientsVoc. If a variant exists, we make the change. If multiple variants are found, we prefer not to correct the token.


Both methods are complementary and rely heavily on the quality of the vocabularies.

## Pipeline Model

The Pipeline model is an abstraction that enable chained-algorithms to be tested against the dataset. It became clear that the best model would be a combination of different algorithms to deal with the different types of errors.

The Pipeline model takes as input a list of models and apply prediction by chaining predictions from them.

# Performances

## Metrics

Several generic metrics are computed to compare models :
- **number_items** : total number of items tested.
- **number_should_have_been_changed** : number of items that needs a correction .
- **number_changed** : number of items that has at least 1 modification.
- **number_correct_when_changed** : number of changed items that are exactly correct, character by character.
- **txt_precision** : precision of the model using an exact matching. Equals `number_correct_when_changed / number_changed`.
- **txt_recall** : recall of the model using an exact matching. Equals `number_correct_when_changed / number_should_have_been_changed`.
- **txt_similarity** : similarity score between predicted and correct descriptions. Similarity is computed using SequenceMatcher.

It turns out that those metrics were not good enough to assess the performances of the spellchecker. For instance, a model could be penalized because a correct change has been made to an ingredient but the full description was still containing a mistake. Overall, txt_precision is higher on models that does less changes which is not our purpose.


To deal with this problem, we introduced "per ingredient metrics". The main strategy has been to split descriptions into lists of ingredients and compare those lists. Metrics are computed using 3 lists of ingredients : the original list, the predicted list and the correct list. The idea is to compare those lists to determine, for example, how many ingredients are simultaneously in predicted list and correct list but not original list. Different approaches has been tested (using sets, Counters and Sequence matching). The current one is the most advanced. It uses Sequence Matching from Python's Difflib library, applied on the lists. We struggled to use it because of the non-symmetrical aspect of the algorithm. We made it artificially symmetrical by wrapping it into a process that computes all possible combinations and take the best one. Metrics are :
- **ingr_recall** : How many correct ingredients that were not in original description have been predicted ?
- **ingr_precision** : How many predicted ingredients that were not in original description are correct ?
- **ingr_fidelity** : How many correct ingredients from the original description are still correct in prediction ?

Those metrics can be computed on an item per item basis. Micro and Macro averages are computed for global metrics.

## FR dataset

Another review of the performances is also available [here](https://docs.google.com/spreadsheets/d/1iuc3O_6oSvnKM9uHNpRD1dTd8wXESVXdlEOCct2QcGY/edit#gid=0).

| Model name | IdentityModel | Percentages | Replacements | Vocabulary | Pipeline Percentages > Replacements > Vocabulary | RobotOFF (index product, confidence 1) | **Pipeline Replacements > Percentages > RobotOFF (index product_all, confidence 0.5) > Vocabulary** |
|---|---|---|---|---|---|---|---|
| Nb of items | 760 | 760 | 760 | 760 | 760 | 760 | **760** |
| Nb should have changed | 389 | 389 | 389 | 389 | 389 | 389 | **389** |
| Nb changed | 0 | 198 | 15 | 36 | 234 | 84 | **288** |
| Nb correct when changed | 0 | 130 | 4 | 10 | 146 | 16 | **162** |
| Txt similarity | 98,51%  | 98,67% | 98,58% | 98,57% | 99,96% | 98,75% | **99,06%** |
| Macro precision | None | 86,62% | 94,44% | 88,52% | 87,78% | 85,99% | **85,53%** |
| Macro recall  | None | 15,67% | 2,17% | 6,88% | 24,71% | 17,20% | **41,40%** |
| Macro fidelity | 100% | 99,96% | 100% | 99,996% | 99,96% | 100% | **99,95%** |
