import pandas as pd
from utils import load_dataset
from constants import FR_TEST_SET_PATH

from models.perfect import PerfectModel
from models.identity import IdentityModel
from evaluation.metrics import evaluation_metrics, format_ingredients

models = [
    PerfectModel(),
    IdentityModel(),
]

items = load_dataset(FR_TEST_SET_PATH)
valid_items = [item for item in items if 'VALID' in item['tags']]

results = pd.DataFrame({
    model.__class__.__name__: evaluation_metrics(valid_items, model.predict(valid_items))
    for model in models
})

print(results)
