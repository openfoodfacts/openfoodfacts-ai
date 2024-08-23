import pytest

from spellcheck.processing import DataProcessor


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            (
                "patata 15%, piment 15,2%, pamplemousse 47 %, aubergine 18.1 %",
                "patata 15 %, piment 15,2 %, pamplemousse 47 %, aubergine 18.1%"       
            ),
            "patata 15%, piment 15,2%, pamplemousse 47 %, aubergine 18.1 %"
        ),
        (
            (
                "escargot 14'%, olives 28 %",
                "escargot 14%, olives 28%"
            ),
            "escargot 14%, olives 28 %",
        ),
        (
            (
                "farine 14&'!%, ravioli 47%",
                "farine 14 %, ravioli 47 %"
            ),
            "farine 14 %, ravioli 47%",
        ),
        (
            (
                "farine 14 %, ravioli 47%",
                "farine 14'%, raviolo 47 %"
            ),
            "farine 14'%, raviolo 47%"
        ),
    ]
)
def test_align_whitespace_percentage(inputs, expected):
    reference, text = inputs
    aligned_text = DataProcessor._align_whitespace_percentage(
        reference=reference, 
        text=text
    )
    assert aligned_text == expected


@pytest.mark.parametrize(
       "inputs, expected",
    [
        (
            (
                "oeuf, bœuf",
                "œuf, boeuf"       
            ),
            "oeuf, bœuf"
        ),
        (
            (
                "œuf, boeuf",
                "oeuf, bœuf"
            ),
            "œuf, boeuf"
        ),
        (
            (
                "bœuf, œuf, ...",
                "boeuf, œuf, ..."
            ),
            "bœuf, œuf, ..."
        ),
        # NOTE: Doesn't work since I cannot detect a non-match. Is not a priority for now
        # (
        #     (
        #         "buf, œuf",
        #         "boeuf, œuf"
        #     ),
        #     "boeuf, œuf"        
        # ),
    ] 
)
def test_align_oe(inputs, expected):
    reference, text = inputs
    aligned_text = DataProcessor._align_oe(
        reference=reference, 
        text=text
    )
    assert aligned_text == expected
