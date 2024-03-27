import pytest

from evaluation.metrics import per_item_ingredients_metrics


@pytest.mark.parametrize(
    "original,correct,prediction,output",
    [
        (
            "graine de ciurge",
            "graine de courge",
            "graine de ciurge",
            {
                "precision_num": 0,
                "precision_den": 0,
                "recall_num": 0,
                "recall_den": 1,
                "precision": None,
                "recall": 0.0,
            },
        ),
        (
            "graine de ciurge",
            "graine de courge",
            "graine de courge",
            {
                "precision_num": 1,
                "precision_den": 1,
                "recall_num": 1,
                "recall_den": 1,
                "precision": 100.0,
                "recall": 100.0,
            },
        ),
        (
            "graine de ciurge grilée",
            "graine de courge grillée",
            "graine de courge grillée",
            {
                "precision_num": 2,
                "precision_den": 2,
                "recall_num": 2,
                "recall_den": 2,
                "precision": 100.0,
                "recall": 100.0,
            },
        ),
        (
            "graine de ciurge grilée",
            "graine de courge grillée",
            "graine de courge graile",
            {
                "precision_num": 1,
                "precision_den": 2,
                "recall_num": 1,
                "recall_den": 2,
                "precision": 50.0,
                "recall": 50.0,
            },
        ),
        (
            "fqrine deblé malté",
            "farine de blé malté",
            "farine deblé malté",
            {
                "precision_num": 1,
                "precision_den": 1,
                "recall_num": 1,
                "recall_den": 3,
                "precision": 100.0,
                "recall": 100.0 / 3,
            },
        ),
        (
            "fqrine deblé malté",
            "farine de blé malté",
            "farine de blé malté",
            {
                "precision_num": 3,
                "precision_den": 3,
                "recall_num": 3,
                "recall_den": 3,
                "precision": 100.0,
                "recall": 100.0,
            },
        ),
    ],
)
def test_per_item_ingredients_metrics_test(
    original: str, correct: str, prediction: str, output
):
    o = per_item_ingredients_metrics(original, correct, prediction)
    for k, v in output.items():
        print(o)
        assert o[k] == v
