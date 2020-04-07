import time
import pandas as pd
from utils import load_dataset
from paths import FR_TEST_SET_PATH

from models.regex import RegexModel
from models.perfect import PerfectModel
from models.identity import IdentityModel
from evaluation.metrics import evaluation_metrics, format_ingredients

models = [
    # PerfectModel(),
    IdentityModel(),
    RegexModel('percentages'),
    RegexModel('replacements'),
]

items = load_dataset(FR_TEST_SET_PATH)
valid_items = [item for item in items if 'VALID' in item['tags']]

results_dict = {}
for model in models:
    model_name = model.name
    t0 = time.time()
    results_dict[model_name] = evaluation_metrics(valid_items, model.predict_save(valid_items))
    t1 = time.time()
    results_dict[model_name]['time_elapsed (s)'] = round(t1-t0, 4)

results = pd.DataFrame(results_dict)

print(results)
