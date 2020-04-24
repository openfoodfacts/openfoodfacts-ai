import time
import json
from utils import load_dataset
from paths import FR_TEST_SET_PATH, METRICS_DF_FILENAME

from models.regex import RegexModel
from models.perfect import PerfectModel
from models.identity import IdentityModel
from models.robotoff import RobotoffAPIModel
from models.pipeline import PipelineModel

from evaluation.metrics import Evaluation

models = [
    # PerfectModel(),
    # IdentityModel(),
    # RegexModel(),
    # RegexModel("percentages"),
    # RegexModel("replacements"),
    # RegexModel("vocabulary"),
    # RobotoffAPIModel(index="product", confidence=0.5),
    # RobotoffAPIModel(index="product", confidence=2),
    # RobotoffAPIModel(index="product_extended", confidence=1),
    # PipelineModel(
    #     models=[
    #         RegexModel("replacements"),
    #         RobotoffAPIModel(index="product", confidence=1),
    #     ]
    # ),
    # PipelineModel(
    #     models=[
    #         RegexModel("replacements"),
    #         RobotoffAPIModel(index="product", confidence=1),
    #         RegexModel("vocabulary"),
    #     ]
    # ),
    # RobotoffAPIModel(index="product", confidence=1),
    # RobotoffAPIModel(index="product_all", confidence=1),
    # RobotoffAPIModel(index="product_extended", confidence=1),
    # RobotoffAPIModel(index="product_extended_all", confidence=1),
    # RobotoffAPIModel(index="product_all", confidence=5),
    # RobotoffAPIModel(index="product_all", confidence=10),
    # RobotoffAPIModel(index="product_all", confidence=20),
    # RobotoffAPIModel(index="product_all", confidence=50),
    # RobotoffAPIModel(index="product_all", confidence=100),
    # RobotoffAPIModel(index="product_all", confidence=500),
    # RobotoffAPIModel(index="product", confidence=500),
    # RobotoffAPIModel(index="product", confidence=500),
    # RobotoffAPIModel(index="product", confidence=1000),
    # RobotoffAPIModel(index="product", confidence=5000),
    # RobotoffAPIModel(index="product", confidence=10000),
    # RobotoffAPIModel(index="product", confidence=50000),
    # RobotoffAPIModel(index="product", confidence=100000),
]

items = load_dataset(FR_TEST_SET_PATH)
valid_items = [item for item in items if item["tags"] == ["VALID"]]

for model in models:
    t0 = time.time()
    predictions = model.predict_save(valid_items)
    evaluation = Evaluation(valid_items, predictions)
    metrics = evaluation.metrics()
    df = Evaluation(valid_items, predictions).detailed_dataframe()
    t1 = time.time()

    metrics["time_elapsed (s)"] = round(t1 - t0, 4)

    with (model.last_experiment_path / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    evaluation.detailed_dataframe().to_csv(
        model.last_experiment_path / METRICS_DF_FILENAME, index=False
    )

    print("\n" + "-" * 60)
    print(f"Model : {model.name}")
    for key, value in metrics.items():
        print(f"{key:33} : {value}")
