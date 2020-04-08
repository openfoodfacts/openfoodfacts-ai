import time
from utils import load_dataset
from paths import FR_TEST_SET_PATH

from models.regex import RegexModel
from models.perfect import PerfectModel
from models.identity import IdentityModel
from models.robotoff import RobotoffAPIModel
from models.pipeline import PipelineModel
from evaluation.metrics import evaluation_metrics, format_ingredients

models = [
    PerfectModel(),
    IdentityModel(),
    RegexModel(),
    RegexModel('percentages'),
    RegexModel('replacements'),
    RobotoffAPIModel(),
    PipelineModel(
        models=[
            RegexModel(),
            RobotoffAPIModel(),
        ]
    ),
]

items = load_dataset(FR_TEST_SET_PATH)
valid_items = [item for item in items if item['tags'] == ['VALID']]

results_dict = {}
for model in models:
    model_name = model.name
    t0 = time.time()
    results_dict[model_name] = evaluation_metrics(valid_items, model.predict_save(valid_items))
    t1 = time.time()
    results_dict[model_name]['time_elapsed (s)'] = round(t1-t0, 4)

for model_name, metrics in results_dict.items():
    print('\n' + '-' * 60)
    print(f'Model : {model_name}')
    for key, value in metrics.items():
        print(f'{key:33} : {value}')
