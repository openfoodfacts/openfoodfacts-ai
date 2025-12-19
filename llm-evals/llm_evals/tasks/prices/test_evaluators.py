import pytest

from llm_evals.tasks.prices.price_tag_extraction.evaluators import (
    generate_pluralization_forms,
    generate_pluralization_forms_single_word,
)


@pytest.mark.parametrize(
    "input_category, expected_forms",
    [
        ("Kakis", {"Kakis", "Kaki"}),
        ("Broccolis", {"Broccolis", "Broccoli"}),
        ("Berries", {"Berries", "Berry"}),
        ("Leaves", {"Leaves", "Leaf", "Leafe"}),
        ("Tomatoes", {"Tomatoes", "Tomato"}),
        ("Cherries", {"Cherries", "Cherry"}),
        ("Cities", {"Cities", "City"}),
        ("Potatoes", {"Potatoes", "Potato"}),
    ],
)
def test_generate_singular_forms(input_category, expected_forms):
    singular_forms = generate_pluralization_forms_single_word(input_category)
    assert singular_forms == expected_forms


@pytest.mark.parametrize(
    "input_category, expected_forms",
    [
        ("Fresh Apples", {"Fresh Apples", "Fresh Apple"}),
        ("Red Berries", {"Red Berries", "Red Berry"}),
        ("Green Leaves", {"Green Leaves", "Green Leaf", "Green Leafe"}),
        ("Large Tomatoes", {"Large Tomatoes", "Large Tomato"}),
        ("Small Cherries", {"Small Cherries", "Small Cherry"}),
        ("Rice wines", {"Rice wines", "Rice wine"}),
    ],
)
def test_generate_pluralization_forms(input_category, expected_forms):
    forms = generate_pluralization_forms(input_category)
    assert forms == expected_forms
