import pytest

from utils.evaluation import SpellcheckEvaluator


ORGINALS = [
    "cacao maigre en Sucre poudre 20% - émulsifiant : léci - thines de tournesol - carbo - nate de magnésium",
    "Ananas, Ananassaft, Säuerungs - mittel: Citronensäure",
    "_Cacahuetes_ con cáscara tostado. _Trazas de frutos de cáscara_.",
    "The cas is on the firdge"
]
REFERENCES = [
    "cacao maigre en Sucre poudre 20% - émulsifiant : lécithines de tournesol - carbonate de magnésium",
    "Ananas, Ananassaft, Säuerungsmittel: Citronensäure",
    "_Cacahuetes_ con cáscara tostado. Trazas de frutos de cáscara.",
    "The cat is in the fridge"
]
PREDICTIONS = [
    "cacao maigre en Sucre pdre 20% - émulsifiant : lécithines de tournesol - carbona de magnésium",
    "Ananas, Säuerungsmittel: Citronensäure",
    "Cacahuetes con cáscara tostado. _Trazas de frutos de cáscara_.",
    "The big cat is in the fridge"
    
]

# Init
evaluator = SpellcheckEvaluator(originals=ORGINALS)


def test_evaluate(predictions = PREDICTIONS, references = REFERENCES):
    """Test the overall function evaluate()."""
    evaluator.evaluate(predictions=predictions, references=references)
    assert True


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            (
                [1, 2, 3, 4, 5, 6, 7],
                [1, 5, 2, 3, 4, 7]
            ),
            [(1, 1), (None, 5), (2, 2), (3, 3), (4, 4), (5, None), (6, None), (7, 7)]
        ),
        (
            (
                [1, 2, 3],
                [7, 8, 9, 1, 2, 4, 5, 3]
            ),
            [(None, 7), (None, 8), (None, 9), (1, 1), (2, 2), (None, 4), (None, 5), (3, 3)]
        ),
        (
            (
                [2127, 26997, 11, 1556, 276, 56692, 728, 11, 328, 2357, 8977, 29222, 482, 48432, 301, 25, 18002, 2298, 729, 2357, 554],
                [2127, 26997, 11, 1556, 276, 56692, 728, 11, 328, 2357, 8977, 2234, 3647, 96383, 25, 18002, 2298, 729, 2357, 554]
            ),
            [(2127, 2127), (26997, 26997), (11, 11), (1556, 1556), (276, 276), (56692, 56692), (728, 728), (11, 11), (328, 328), (2357, 2357), (8977, 8977), (29222, None), (482, 2234), (48432, 3647), (301, 96383), (25, 25), (18002, 18002), (2298, 2298), (729, 729), (2357, 2357), (554, 554)]
        ),
        (
            (
                [791, 4865, 374, 389, 279, 282, 2668, 713],
                [791, 8415, 374, 304, 279, 38681]
            ),
            [(791, 791), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, None), (2668, None), (713, 38681)]

        )
    ]
)
def test_sequence_alignment(inputs, expected):
    """Test sequence alignment method."""
    seq1, seq2 = inputs
    alignment = evaluator.sequence_alignment(seq1, seq2)
    assert alignment == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            (
                [(1, 1), (2, 2), (None, 3), (4, 4)],
                [(1, 1), (2, 2), (4, 4)]
            ),
            (
                [(1, 1), (2, 2), (None, 3), (4, 4)],
                [(1, 1), (2, 2), (None, None), (4, 4)]
            )
        ),
        (
            (
                [(1, 1), (2, 2), (4, 4)],
                [(1, 1), (2, 2), (None, 3), (4, 4)]
            ),
            (
                [(1, 1), (2, 2), (None, None), (4, 4)],
                [(1, 1), (2, 2), (None, 3), (4, 4)]
            )
        ),
        (
            (
                [(1, 1), (None, 7), (2, 2), (4, 4)],
                [(1, 1), (2, 2), (None, 3), (4, 4)]
            ),
            (
                [(1, 1), (None, 7), (2, 2), (None, None), (4, 4)],
                [(1, 1), (None, None), (2, 2), (None, 3), (4, 4)]
            )
        ),
        (
            (
                [(791, 791), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, 38681), (2668, None), (713, None)],
                [(791, 791), (4865, 2466), (None, 8415), (374, 374), (389, 304), (279, 279), (282, 38681), (2668, None), (713, None)]
            ),
            (
                [(791, 791), (4865, 8415), (None, None), (374, 374), (389, 304), (279, 279), (282, 38681), (2668, None), (713, None)],
                [(791, 791), (4865, 2466), (None, 8415), (374, 374), (389, 304), (279, 279), (282, 38681), (2668, None), (713, None)]
            )
        )
    ]
)
def test_align_pairs(inputs, expected):
    """Test pairs alignment."""
    pairs1, pairs2 = inputs
    aligned_pairs1, aligned_pairs2 = evaluator.align_pairs(pairs1, pairs2, neutral_pair=(None, None))
    assert aligned_pairs1, aligned_pairs2 == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            (
                [(1, 1), (2, 3), (4, 5), (6, 6)],
                [(1, 1), (2, 2), (4, 7), (6, 6)]
            ),
            [0]
        ),
        (

            (
                [(1, 1), (2, 3), (4, 5), (None, None), (6, 6)],
                [(1, 1), (2, 3), (4, 5), (None, 6), (6, 7)]
            ),
            [1, 1]
        ),
        (
            (
                [(1, 1), (2, 3), (5, 6), (7, 8)],
                [(1, 1), (2, 4), (5, 6), (7, 9)]
            ),
            [0, 1, 0]
        ),

        (
            (
                [(791, 791), (None, None), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, None), (2668, None), (713, 38681)],
                [(791, 791), (None, 2466), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, None), (2668, None), (713, 38681)]
            ),
            [1, 1, 1, 1, 1]
        ),
        (
            (
                [(2127, 2127), (26997, 26997), (11, 11), (1556, 1556), (276, 276), (56692, 56692), (728, 728), (11, 11), (328, 328), (2357, 2357), (8977, 8977), (29222, None), (482, 2234), (48432, 3647), (301, 96383), (25, 25), (18002, 18002), (2298, 2298), (729, 729), (2357, 2357), (554, 554)],
                [(2127, 2127), (26997, 26997), (11, None), (1556, None), (276, None), (56692, None), (728, None), (11, 11), (328, 328), (2357, 2357), (8977, 8977), (29222, None), (482, 2234), (48432, 3647), (301, 96383), (25, 25), (18002, 18002), (2298, 2298), (729, 729), (2357, 2357), (554, 554)]
            ),
            [1, 1, 1, 1]
        )
    ]
)
def test_get_correction_true_positives(inputs, expected):
    """Test correction precison calculation.

    It corresponds to the accuracy of the model to select the right token based only on tokens that
    were supposed to be changed.
    """
    ref_pairs, pred_pairs = inputs
    correction_precision = evaluator.get_correction_true_positives(
        ref_pairs, pred_pairs
    )
    assert correction_precision == expected
    
evaluator.get_correction_true_positives(
    ref_pairs=[(791, 791), (None, None), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, None), (2668, None), (713, 38681)],
    pred_pairs=[(791, 791), (None, 2466), (4865, 8415), (374, 374), (389, 304), (279, 279), (282, None), (2668, None), (713, 38681)]
)