# SpellCheck

OpenFoodFact SpellCheck project aims to correct misspelled ingredients on product descriptions. Those errors mainly appear because of the use of the OCR.  

## Project install

This project runs using Python3.

### Install dependencies

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

### Label data

To label data, you must have a MongoDB running with the OFF database on it. By default, it will connect to localhost/port 27017 without any password. To change this behavior, edit the `mongo.py` file.  
Once MongoDB is configured, run the following command :
```
streamlit run label.py
```
It will open a streamlit dashboard on the browser.

### Tests

```
python test.py
```
It will run tests on the FR dataset, using all available models.
To configure which models to test, edit the `test.py` file.  

Each time the script is ran, a folder is created under `experiment/model_name/datetime/` with the `original.txt`, `correct.txt` and `prediction.txt` files.

### (README) TODO
- explain how to label / why label data.
- explain models
- explain metrics
- explain goal of having test set
