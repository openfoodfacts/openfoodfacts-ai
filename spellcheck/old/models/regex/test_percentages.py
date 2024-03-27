import pytest

from models.regex.percentages import format_percentages

TEST_PATH = "test_sets/percentages/fr.txt"


with open(TEST_PATH, "r") as f:
    test_set = [tuple(item.split("\n")[:2]) for item in f.read().split("\n\n")]


@pytest.mark.parametrize("original, correct", test_set)
def test_format_percentages(original: str, correct: str):
    formatted = format_percentages(original)
    print(formatted)
    assert formatted == correct
