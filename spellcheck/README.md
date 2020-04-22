# SpellCheck

OpenFoodFact SpellCheck project aims to correct misspelled ingredients on product descriptions. Those errors mainly appear because of the use of the OCR.  

## Install dependencies

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

## Label data

To label data, you must have a MongoDB running with the OFF database on it. By default, it will connect to localhost/port 27017 without any password. To change this behavior, edit the `mongo.py` file.  
Once MongoDB is configured, run the following command :
```
streamlit run label.py
```
It will open a streamlit dashboard on the browser.

## Tests

### Run tests

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

### Review

A CLI tool is available to review model predictions once the test pipeline has been run. It takes the path to the experiment folder and an item_id as input. It outputs :
- the original, correct and predicted descriptions with normalization (lowercase, œu->oe,...)
- the original, correct and predicted ingredients Counter, as they are computed for the metrics
- some curated ingredients Counters with only relevant ingredients (those appearing in the calculation of precision/recall/loyalty metrics)

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

### (README) TODO
- explain how to label / why label data.
- explain models
- explain metrics
- explain goal of having test set
