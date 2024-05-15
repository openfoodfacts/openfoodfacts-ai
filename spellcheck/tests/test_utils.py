import pytest

from spellcheck.utils import (
    get_repo_dir, 
    get_logger,
    show_diff
)


DELETED_ELEMENT = "#"


def test_get_repo_dir():
    assert get_repo_dir().name == "spellcheck"
    assert get_repo_dir().is_dir()


def test_get_logger():
    get_logger()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            (
                "hello world",
                "hllo borld"
            ),
            f"h<mark style=background-color:#FCF910>{DELETED_ELEMENT}</mark>llo <mark style=background-color:#FCF910>b</mark>orld"
        ),
    ]
)
def test_show_diff(test_input, expected):
    original, corrected = test_input
    highlighted_correction = show_diff(
        original_text=original,
        corrected_text=corrected,
        deleted_element=DELETED_ELEMENT
    )
    assert highlighted_correction == expected
