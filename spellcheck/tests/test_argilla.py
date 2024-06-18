from typing import Iterable

import pytest

from spellcheck.argilla_modules import (
    ArgillaModule,
    BenchmarkArgilla,
    BenchmarkEvaluationArgilla,
    IngredientsCompleteEvaluationArgilla,
    TrainingDataArgilla
)


ORIGINALS = ["text1", "text2", "text3"]
REFERENCES = ["text1", "text", "text3"]
PREDICTIONS = ["text1", "text2", "text"]
METADATA = [
    {"lang": "fr", "code": 123456789}, 
    {"lang": "en", "code": 123456789}, 
    {"lang": "es", "code": 123456789}
]


@pytest.fixture
def modules() -> Iterable[ArgillaModule]:
    return [
        BenchmarkArgilla(ORIGINALS, REFERENCES, METADATA),
        BenchmarkEvaluationArgilla(ORIGINALS, REFERENCES, PREDICTIONS, METADATA),
        IngredientsCompleteEvaluationArgilla(ORIGINALS, PREDICTIONS, METADATA),
        TrainingDataArgilla(ORIGINALS, REFERENCES, METADATA)
    ]


@pytest.fixture
def evaluation_modules() -> Iterable[ArgillaModule]:
    return [
        BenchmarkEvaluationArgilla(ORIGINALS, REFERENCES, PREDICTIONS, METADATA),
        IngredientsCompleteEvaluationArgilla(ORIGINALS, PREDICTIONS, METADATA)
    ]


def test_modules(modules: Iterable[ArgillaModule]):
    """Test main methods from Argilla Modules."""
    for module in modules:
        print(f"Module name: {module.__class__.__name__}")
        dataset = module._prepare_dataset()
        records = module._prepare_records()
        dataset.add_records(records)


def test_evaluation_highlights(evaluation_modules: Iterable[ArgillaModule]):
    """Test highlighted differences during evaluation"""
    for module in evaluation_modules:
        if hasattr(module, "highlighted_references"):
            assert module.highlighted_references
        if hasattr(module, "highlighted_predictions"):
            assert module.highlighted_predictions
