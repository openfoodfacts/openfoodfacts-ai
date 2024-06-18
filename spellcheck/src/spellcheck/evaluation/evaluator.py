import logging
from abc import ABC, abstractmethod
from typing import Mapping, Tuple, List, Iterable
from tqdm import tqdm

import tiktoken
import numpy as np


LOGGER = logging.getLogger(__name__)


class Evaluator(ABC):
    """Evaluation class."""

    @abstractmethod
    def evaluate(
        self,
        predictions: Iterable[str],
        references: Iterable[str], 
        **kwargs
    ) -> Mapping:
        """Abstract method to calculate metrics based on the predictions and the evaluation method.

        Args:
            predictions (List[str]): Batch of text predictions

        Returns:
            Mapping: Metrics
        """
        raise NotImplementedError


class SpellcheckEvaluator(Evaluator):
    """Evaluator designed for Spellcheck.

    This module is designed to evaluate how well Spellcheck models performs on recognizing errors in the
    list of ingredients and correctly fixing them.

    Therefore, 3 elements are necessary:
        * The original text with typos/errors
        * The expected correction.
        * The prediction from the model.

    The uniqueness of this evaluation algorithm lies in its calculation of precision, recall, and F1 scores specifically for errors to be corrected,
    which are directly extracted from texts.

    The process is divided into 4 steps:
        * Texts (Original-Reference-Prediction) are tokenized using a Byte Pair Encoding (BPE) tokenizer from the `tiktoken` library.
        * Encoded originals and references are aligned using Sequence Alignment technique to locate which tokens were transformed, added, or deleted. 
    The same process is applied to encoded originals and predictions. 
        * Pairs of tokens (Original-Reference; Original-Prediction) are aligned to consider gaps in case Reference and/or Prediction have different length.
        * Precision, Recall, and F1-score are calculated based on pairs of tokens for either the reference and the prediction.

    This module produces two distinct sets of metrics:
        * Metrics related to the indices of the tokens to be corrected:
            * Precision
            * Recall
            * F1
            * F1-beta
        * Metrics related to the model's precision in predicting the correct tokens:
            * Correction precision

    Args:
        originals (List[str]): Batch of original ingredient lists to correct
        references (List[str]): Batch of expected ingredients lists after correction
        encoding_name (str, optional): BPE tokenizer from the tiktoken library. Defaults to "cl100k_base".
        beta (float, optional): Coefficient for F1_beta metric. A coefficient of less than 1.0 gives more weight to the Recall, 
    whereas a coefficient greater than 1.0 gives more weight to the Precision.
        drop_rate (foat, optional): Some predictions are (almost) empty, which means it would be irrelevant to compare them (alignment issue)
    Therefore we drop to not bias the metrics. An additional metric is added to count them. 
    """

    def __init__(
        self,
        originals: Iterable[str],
        encoding_name: str = "cl100k_base",
        beta: float = 1.0,
        drop_rate: float = 0.4
    ) -> None:
        self.originals = originals
        self.encoder = tiktoken.get_encoding(encoding_name=encoding_name)
        self.beta = beta
        self.drop_rate = drop_rate

    def evaluate(self, predictions: List[str], references: List[str]) -> Mapping:
        """Evaluate the performance of Spellcheck on correcting ingredient lists for ingredients extraction.

        Metrics: 
            * Precision
            * Recall
            * F1
            * F1-beta
            * Correction precision (precision on predicting the correct token)

        Args:
            predictions (List[str]): Correction from the Spellcheck
            references (List[str]): Reference

        Raises:
            ValueError: Batch of texts not provided.

        Returns:
            Mapping: Metrics. Keys: [precision, recall, f1, f1_beta, beta]
        """
        if not isinstance(predictions, List):
            raise ValueError(
                f"Evaluate requires a batch of texts. Got {type(predictions)} instead."
            )

        # To calculate Precision Recall across the examples.
        true_positives = []
        false_positives = []
        false_negatives = []
        correction_true_positives = []
        drop_count = 0

        # Batch
        for original, reference, prediction in tqdm(
            zip(self.originals, references, predictions), 
            total=len(predictions),
            desc="Evaluation"
        ):

            # Convert into tokens.
            original_tokens = self.encoder.encode(original)
            reference_tokens = self.encoder.encode(reference)
            prediction_tokens = self.encoder.encode(prediction)

            # Drop short or empty sequences to avoid alignment computation bias
            if len(prediction_tokens) < self.drop_rate * len(reference_tokens):
                drop_count += 1
                pass

            # Align tokens to detect transformation, deletion or addition
            ref_pairs = self.sequence_alignment(original_tokens, reference_tokens)
            pred_pairs = self.sequence_alignment(original_tokens, prediction_tokens)

            # Align ref-pairs and pred-pairs 
            aligned_ref_pairs, aligned_pred_pairs = self.align_pairs(
                ref_pairs, pred_pairs
            )

            # Convert pairs into sparse matrices for metrics calculation
            sparse_ref_pairs = self.convert_pairs_into_sparse(aligned_ref_pairs)
            sparse_pred_pairs = self.convert_pairs_into_sparse(aligned_pred_pairs)
            assert len(sparse_ref_pairs) == len(sparse_pred_pairs), "Ref and pred pairs don't have the same length!"

            inverse_sparse_ref_pairs = [1 if i == 0 else 0 for i in sparse_ref_pairs]
            inverse_sparse_pred_pairs = [1 if i == 0 else 0 for i in sparse_pred_pairs]

            seq_true_positives = np.multiply(sparse_ref_pairs, sparse_pred_pairs)
            seq_false_positives = np.multiply(inverse_sparse_ref_pairs, sparse_pred_pairs)
            seq_false_negatives = np.multiply(sparse_ref_pairs, inverse_sparse_pred_pairs)

            # Also check if model token predictions are correct
            seq_correction_true_positives = self.get_correction_true_positives(
                ref_pairs=aligned_ref_pairs,
                pred_pairs=aligned_pred_pairs
            )

            true_positives.extend(seq_true_positives)
            false_positives.extend(seq_false_positives)
            false_negatives.extend(seq_false_negatives)
            correction_true_positives.extend(seq_correction_true_positives)

        # Sum
        true_positive = np.sum(true_positives)
        false_positive = np.sum(false_positives)
        false_negative = np.sum(false_negatives)
        correction_true_positive = np.sum(correction_true_positives)

        # Metrics calculation
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall) if (precision + recall) != 0 else 0
        correction_precision = correction_true_positive / (true_positive + false_positive) if true_positive != 0 else 0
        correction_recall = correction_true_positive / (true_positive + false_negative) if correction_true_positive !=0 else 0

        # Mean results for the entire batch 
        results = {
            "correction_precision": correction_precision,
            "correction_recall": correction_recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_beta": f1_beta,
            "beta": self.beta,
            "drop_count": drop_count
        }
        LOGGER.info(f"Evaluation metrics: {results}")
        return results

    @staticmethod
    def sequence_alignment(
        tokens_1: List[int],
        tokens_2: List[int],
        match_score: int = 1,
        mismatch_score: int = -1,
        gap_penalty: int = -2,
    ) -> List[Tuple]:
        """Needleman-Wunsch Sequence alignment algorithm used in bioinformatique to align 2 sequences.
        Wiki: https://en.wikipedia.org/wiki/Sequence_alignment
        Needleman-Wunsch algorithm: https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm

        The algorithm pairs tokens that were added, deleted or transformed after modification.

        Args:
            tokens_1 (List[int]): Tokenized representation of the first text.
            tokens_2 (List[int]): Tokenized representation of the second text.
            match_score (int, optional): Penalize or reward depending on a low or high score.
            mismatch_score (int, optional): Penalize or reward depending on a low or high score.
            gap_penalty (int, optional): Penalize or reward depending on a low or high score.

        Returns:
            List[Tuple]: List of token pairs.

        Example 1:    
        ```
        tokens_1 = [791, 8415, 4502, 389, 279, 282, 1425, 13]
        tokens_2 = [791, 8415, 374, 389, 279, 38681, 13]

        alignment = [(791, 791), (8415, 8415), (4502, 374), (389, 389), (279, 279), (282, None), (1425, 38681), (13, 13)]
        ```
        
        Example 2:
        ```
        tokens_1 = [54, 51, 23, 165, 415, 61, 561]
        tokens_2 = [54, 51, 906, 23, 165, 415, 61, 4100]

        alignment = [(54, 54), (51, 51), (None, 906), (23, 23), (165, 165), (None, 165), (415, 415), (61, 61), (561, 4100)]
        ```
        """
        # Initialize matrix
        matrix = [[i * gap_penalty] + [0] * (len(tokens_2)) for i in range(len(tokens_1) + 1)]
        matrix[0] = [(j * gap_penalty) for j in range(len(tokens_2) + 1)]
        # Fill in the matrix
        for i in range(1, len(tokens_1) + 1):
            for j in range(1, len(tokens_2) + 1):
                match = matrix[i - 1][j - 1] + (
                    match_score
                    if tokens_1[i - 1] == tokens_2[j - 1]
                    else mismatch_score
                )
                delete = matrix[i - 1][j] + gap_penalty
                insert = matrix[i][j - 1] + gap_penalty
                matrix[i][j] = max(match, delete, insert)
        # Traceback
        alignment = []
        i, j = len(tokens_1), len(tokens_2)
        while i > 0 and j > 0:
            if matrix[i][j] == matrix[i - 1][j - 1] + (
                match_score if tokens_1[i - 1] == tokens_2[j - 1] else mismatch_score
            ):
                alignment.append((tokens_1[i - 1], tokens_2[j - 1]))
                i -= 1
                j -= 1
            elif matrix[i][j] == matrix[i - 1][j] + gap_penalty:
                alignment.append((tokens_1[i - 1], None))  # Mark deletion
                i -= 1
            elif matrix[i][j] == matrix[i][j - 1] + gap_penalty:
                alignment.append((None, tokens_2[j - 1]))  # Mark insertion
                j -= 1
            

        # Handle remaining elements if any
        while i > 0:
            alignment.append((tokens_1[i - 1], None))  # Mark remaining deletions
            i -= 1
        while j > 0:
            alignment.append((None, tokens_2[j - 1]))  # Mark remaining insertions
            j -= 1
        alignment.reverse()  # Reverse alignment to get correct order
        return alignment

    @staticmethod
    def convert_pairs_into_sparse(pairs: List[Tuple]) -> List[int]:
        """Convert alignement pairs/tuples into a sparse vector. 
        If there is a mismatch between tokens from the same pair, it is considered as a modification (=1). 
        
        Example:
        ```
        pairs = [(791, 791), (8415, 8415), (4502, 374), (389, 389), (279, 279), (282, None), (1425, 38681), (13, 13)]
        sparse_pairs = [0, 0, 1, 0, 0, 1, 1, 0] 
        ```
        Args:
            pairs (List[Tuple]): Iterable of token pairs from the Sequence alignment algorithm.

        Return:
            (List[int]): Sparse vectors.
        """
        return [0 if i == j else 1 for i, j in pairs]
    
    @staticmethod
    def align_pairs(
        pairs1: List[Tuple], 
        pairs2: List[Tuple],
        neutral_pair: Tuple = (None, None)
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """SInce we compare Pairs between the Reference and the Prediction, it's possible that tokens were added or deleted
        in one but not in the other. This leads to a misalignment between pairs of tokens required for calculating 
        Precision and Recall.

        For this reason, we add a "neutral" pair of tokens for each gap in the opposite list of pairs. This "neutral" pair
        aligns the modifed tokens without adding any influence on the metrics calculation.

        Example:
        ```
        Before:
        Orig-Ref pairs: [(400, 400), (350, 350), (20, 18), (21, 40), (None, 51), (23, 23)]
        Orig-Pred pairs: [(400, 400), (350, 350), (None, 800), (20, 18), (21, 40), (23, 80)]
        
        sparse Ref: [0, 0, 1, 1, 1, 0]
        sparse Pred: [0, 0, 1, 1, 1, 1]

        After:
        Orig-Ref pairs: [(400, 400), (350, 350), (None, None), (20, 18), (21, 40), (None, 51) (23, 23)]
        Orig-Pred pairs: [(400, 400), (350, 350), (None, 800), (20, 18), (21, 40), (None, None), (23, 80)] 

        sparse Ref: [0, 0, 0, 1, 1, 1, 0]
        sparse Pred: [0, 0, 1, 1, 1, 0, 1]

        Args:
            pairs1 (List[Tuple]): Can be list of Orig-Ref token pairs
            pairs2 (List[Tuple]): Can be list of Orig-Pred token pairs
            neutral_pairs (Tuple, optional): Pair to insert for alignment. Defaults to (None, None).

        Returns:
            Tuple[List[Tuple], List[Tuple]]: Aligned list of pairs. 
        """
        # Since we insert into the list, we create copies to avoid the global modification
        pairs1_bis, pairs2_bis = pairs1.copy(), pairs2.copy()
        pairs1_gap_ids = [idx for idx, pair in enumerate(pairs1_bis) if pair[0] is None]
        pairs2_gap_ids = [idx for idx, pair in enumerate(pairs2_bis) if pair[0] is None]
        # Insert neutral pair. Reversed to prevent shifting.
        for pred_gap_idx in reversed(pairs2_gap_ids):
            pairs1_bis.insert(pred_gap_idx, neutral_pair)
        for ref_gap_idx in reversed(pairs1_gap_ids):
            pairs2_bis.insert(ref_gap_idx, neutral_pair)
        return pairs1_bis, pairs2_bis

    def get_correction_true_positives(
        self,
        ref_pairs: List[Tuple],
        pred_pairs: List[Tuple], 
    ) -> float:
        """Correction true positives corresponding to the precision of the model the predict the correct token.

        Note:
        We consider only tokens that were modified by the model & were supposed to be modified. 
        It means that if the model missed a token correction, or added one, it is not considered in the correction precision calculation
        in case the token wasn"t supposed to be corrected.

        Args:
            ref_pairs (List[Tuple]): List of Orig-Ref token pairs
            pred_pairs (List[Tuple]): List of Orig-Pred token pairs

        Returns:
            float: precision of picking the right token.
        """
        sparse_ref_pairs, sparse_pred_pairs = (
            self.convert_pairs_into_sparse(ref_pairs), 
            self.convert_pairs_into_sparse(pred_pairs)
        )
        true_positives = np.multiply(sparse_ref_pairs, sparse_pred_pairs)
        correction_true_positives = [
            int(ref_pairs[idx][1] == pred_pairs[idx][1]) 
            for idx, tp in enumerate(true_positives) 
            if tp == 1
        ]
        return correction_true_positives


if __name__ == "__main__":

    # DEBUG
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

    spellcheck_evaluator = SpellcheckEvaluator(originals=ORGINALS)
    results = spellcheck_evaluator.evaluate(predictions=PREDICTIONS, references=REFERENCES)
    print(results)
