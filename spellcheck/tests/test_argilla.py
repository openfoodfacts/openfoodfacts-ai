from typing import Iterable

import pytest

from spellcheck.argilla.deployment import (
    ArgillaModule,
    BenchmarkArgilla,
    BenchmarkEvaluationArgilla,
    IngredientsCompleteEvaluationArgilla,
    TrainingDataArgilla
)
from spellcheck.argilla.extraction import SpellcheckExtraction


ORIGINALS = ["text1", "text2", "text3"]
REFERENCES = ["text1", "text", "text3"]
PREDICTIONS = ["text1", "text2", "text"]
METADATA = [
    {"lang": "fr", "code": 123456789}, 
    {"lang": "en", "code": 123456789}, 
    {"lang": "es", "code": 123456789}
]

ARGILLA_EXTRACTION_EXAMPLE_1 = {
    "url": "https://world.openfoodfacts.org/product/5942262001416",
    "original": "water:snow",
    "reference": [{"user_id": "dfb71753-1187-45e1-8006-629bef2b49e0", "value": "water:snow", "status": "discarded"}],
    "reference-suggestion": "water:snow",
    "reference-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "is_truncated": [],
    "is_truncated-suggestion": None,
    "is_truncated-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "external_id": None,
    "metadata": '{"lang": "ro", "data_origin": "50-percent-unknown"}',
}
ARGILLA_EXTRACTION_EXAMPLE_2 = {
    "url": "https://world.openfoodfacts.org/product/5942262001416",
    "original": "water:snow",
    "reference": [],
    "reference-suggestion": "water:snow",
    "reference-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "is_truncated": [],
    "is_truncated-suggestion": None,
    "is_truncated-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "external_id": None,
    "metadata": '{"lang": "ro", "data_origin": "50-percent-unknown"}',
}
ARGILLA_EXTRACTION_EXAMPLE_3 = {
    "url": None,
    "original": "Ananas, Ananassaft, Säuerungs - mittel: Citronensäure",
    "reference": [
        {
        "user_id": "dfb71753-1187-45e1-8006-629bef2b49e0", 
        "value": "Ananas, Ananassaft, Säuerungsmittel: Citronensäure", 
        "status": "submitted"
        }
    ],
    "reference-suggestion": "Ananas, Ananassaft, Säuerungsmittel: Citronensäure",
    "reference-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "is_truncated": [{"user_id": "dfb71753-1187-45e1-8006-629bef2b49e0", "value": "NO", "status": "submitted"}],
    "is_truncated-suggestion": None,
    "is_truncated-suggestion-metadata": {"type": None, "score": None, "agent": None},
    "external_id": None,
    "metadata": '{"lang": "de", "data_origin": "labeled_data"}',
}


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


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            (ARGILLA_EXTRACTION_EXAMPLE_1, ["submitted"]),
            False
        ),
        (
            (ARGILLA_EXTRACTION_EXAMPLE_2, ["submitted", "pending"]),
            True
        ),
        (
            (ARGILLA_EXTRACTION_EXAMPLE_3, ["submitted"]),
            True
        )
    ]
)
def test_argilla_spellcheck_extraction_filter(inputs, expected):
    """Test Argilla dataset extraction filtering."""
    element, status = inputs
    is_kept = SpellcheckExtraction(
        dataset_name="test_dataset",
        extracted_status=status
    )._filter_fn(element)
    assert is_kept == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            ARGILLA_EXTRACTION_EXAMPLE_3,
            {
                "original": "Ananas, Ananassaft, Säuerungs - mittel: Citronensäure",
                "reference": "Ananas, Ananassaft, Säuerungsmittel: Citronensäure",
                "lang": "de",
                "data_origin": "labeled_data",
                "is_truncated": 0,
            }
        ),
        (
            ARGILLA_EXTRACTION_EXAMPLE_2,
            {
                "original": "water:snow",
                "reference": "water:snow",
                "lang": "ro",
                "data_origin": "50-percent-unknown",
                "is_truncated": 0,
            }
        ),
    ]
)
def test_argilla_spellcheck_extraction_map(inputs, expected):
    """Test Argilla dataset extraction mapping."""
    element = inputs
    extracted = SpellcheckExtraction(
        dataset_name="test_dataset",
        extracted_status=["submitted"]
    )._map_fn(element)
    assert extracted == expected
