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


### (README) TODO
- explain how to label / why label data.
- explain models
- explain metrics
- explain goal of having test set
