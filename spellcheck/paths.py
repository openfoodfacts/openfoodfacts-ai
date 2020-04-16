from pathlib import Path
from datetime import datetime

EXPERIMENT_PATH = Path('experiments')

FR_TEST_SET_PATH = Path('test_sets/fr/uniform_sampling')

METRICS_DF_FILENAME = 'detailed_metrics.csv'


def now():
    return datetime.now().strftime('%Y_%m_%d_%Hh%Mm%Ss')


def new_experiment_path(model_name):
    path = EXPERIMENT_PATH / model_name / now()
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_name_from_path(experiment_path):
    return Path(experiment_path).parent.name
