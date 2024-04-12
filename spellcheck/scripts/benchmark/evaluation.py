import logging
from abc import ABC, abstractmethod
from typing import Mapping, Tuple, List, Iterable

import tiktoken
import numpy as np


LOGGER = logging.getLogger(__name__)


class Evaluator(ABC):
    """Evaluation class."""

    @abstractmethod
    def evaluate(self, predictions, **kwargs) -> Mapping:
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
    """

    def __init__(
        self,
        originals: Iterable[str],
        references: Iterable[str],
        encoding_name: str = "cl100k_base",
        beta: float = 1.0
    ) -> None:
        self.originals = originals
        self.references = references
        self.encoder = tiktoken.get_encoding(encoding_name=encoding_name)
        self.beta = beta

    def evaluate(self, predictions: List[str]) -> Mapping:
        """Evaluate the performance of Spellcheck on correcting ingredient lists for ingredients extraction.

        Metrics: 
            * Precision
            * Recall
            * F1
            * F1-beta
            * Correction precision (precision on predicting the correct token)

        Args:
            predictions (List[str]): Correction from the Spellcheck

        Raises:
            ValueError: Batch of texts not provided.

        Returns:
            Mapping: Metrics. Keys: [precision, recall, f1, f1_beta, beta]
        """
        if not isinstance(predictions, List):
            raise ValueError(
                f"Evaluate requires a batch of texts. Got {type(predictions)} instead."
            )
        
        # Metrics to output for each line
        precisions = []
        recalls = []
        f1s = []
        f1_betas = []
        correction_precisions = []

        # Batch
        for original, reference, prediction in zip(self.originals, self.references, predictions):

            # Convert into tokens.
            original_tokens = self.encoder.encode(original)
            reference_tokens = self.encoder.encode(reference)
            prediction_tokens = self.encoder.encode(prediction)

            # Align tokens to detect transformation, deletion or addition
            ref_pairs = self.sequence_alignment(original_tokens, reference_tokens)
            pred_pairs = self.sequence_alignment(original_tokens, prediction_tokens)

            # Align ref-pairs and pred-pairs 
            aligned_ref_pairs, aligned_pred_pairs = self.align_ref_pred_pairs(
                ref_pairs=ref_pairs,
                pred_pairs=pred_pairs
            )

            # Convert pairs into sparse matrices for metrics calculation
            sparse_ref_pairs = self.convert_pairs_into_sparse(aligned_ref_pairs)
            sparse_pred_pairs = self.convert_pairs_into_sparse(aligned_pred_pairs)
            assert len(sparse_ref_pairs) == len(sparse_pred_pairs), "Ref and pred pairs don't have the same length!"

            inverse_sparse_ref_pairs = [1 if i == 0 else 0 for i in sparse_ref_pairs]
            inverse_sparse_pred_pairs = [1 if i == 0 else 0 for i in sparse_pred_pairs]

            # Calculate metrics
            true_positive = np.sum(np.multiply(sparse_ref_pairs, sparse_pred_pairs))
            false_postive = np.sum(np.multiply(inverse_sparse_ref_pairs, sparse_pred_pairs))
            false_negative = np.sum(np.multiply(sparse_ref_pairs, inverse_sparse_pred_pairs))

            precision = true_positive / (true_positive + false_postive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_beta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            f1_betas.append(f1_beta)

            # Also calculate if model token predictions are correct
            correction_precision = self.compute_correction_precision(
                ref_pairs=aligned_ref_pairs,
                pred_pairs=aligned_pred_pairs,
                sparse_ref_pairs=sparse_ref_pairs, 
                sparse_pred_pairs=sparse_pred_pairs
            )
            correction_precisions.append(correction_precision)

        # Mean results for the entire batch 
        results = {
            "correction_precision": np.mean(correction_precisions),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
            "f1_beta": np.mean(f1_betas),
            "beta": self.beta,
        }
        LOGGER.info(f"Evaluation metrics: {results}")
        return results

    @staticmethod
    def sequence_alignment(
        tokens_1: List[int],
        tokens_2: List[int],
        match_score: int = 2,
        mismatch_score: int = -2,
        gap_penalty: int = -1,
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
        matrix = [[0] * (len(tokens_2) + 1) for _ in range(len(tokens_1) + 1)]
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
            else:
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
    def align_ref_pred_pairs(
        ref_pairs: List[Tuple], 
        pred_pairs: List[Tuple],
        insert_pairs: Tuple = (None, None)
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Reference and Prediction pairs are aligned in case of different lengths to enable metrics calculation.
        In case of one list of pairs smaller than the other one, the neutral pair (None, None) is inserted into the shortest list to 
        be considered as 0 in it's respective sparse vector, which means it doesn't count as a modification.

        Example:
        ```
        Before:
        Orig-Ref pairs: [(400, 400), (350, 350), (20, 18), (21, 40), (23, 23)]
        Orig-Pred pairs: [(400, 400), (350, 350), (None, 800), (20, 18), (21, 40), (23, 80)]
        
        sparse Ref: [0, 0, 1, 1, 0]
        sparse Pred: [0, 0, 1, 1, 1, 1]

        After:
        Orig-Ref pairs: [(400, 400), (350, 350), (None, None), (20, 18), (21, 40), (23, 23)] => len
        Orig-Pred pairs: [(400, 400), (350, 350), (None, 800), (20, 18), (21, 40), (23, 80)] 

        sparse Ref: [0, 0, 0, 1, 1, 0]
        sparse Pred: [0, 0, 1, 1, 1, 1]

        Args:
            ref_pairs (List[Tuple]): List of Orig-Ref token pairs
            pred_pairs (List[Tuple]): List of Orig-Pred token pairs
            insert_pairs (Tuple, optional): Pair to insert for alignment. Defaults to (None, None).

        Returns:
            Tuple[List[Tuple], List[Tuple]]: Aligned list of pairs.
        """
        if len(ref_pairs) != len(pred_pairs):
            longest_pairs = max(ref_pairs, pred_pairs, key=len)
            shortest_pairs = min(ref_pairs, pred_pairs, key=len)
            ids = [idx for idx, pairs in enumerate(longest_pairs) if pairs[0] is None]
            aligned_shortest_pairs = []
            for idx, elt in enumerate(shortest_pairs):
                if idx in ids:
                    # Insert
                    aligned_shortest_pairs.append(insert_pairs)
                    aligned_shortest_pairs.append(elt)
                else:
                    aligned_shortest_pairs.append(elt)
            if ref_pairs is shortest_pairs:
                return aligned_shortest_pairs, longest_pairs
            elif pred_pairs is shortest_pairs:
                return longest_pairs, aligned_shortest_pairs
        if len(ref_pairs) == len(pred_pairs):
            # Do nothing
            return ref_pairs, pred_pairs

    @staticmethod
    def compute_correction_precision(
        ref_pairs: List[Tuple],
        pred_pairs: List[Tuple], 
        sparse_ref_pairs: List[int], 
        sparse_pred_pairs: List[int]
    ) -> float:
        """Correction precision metric correspond to the precision of the model the predict the correct token.

        Args:
            ref_pairs (List[Tuple]): List of Orig-Ref token pairs
            pred_pairs (List[Tuple]): List of Orig-Pred token pairs
            sparse_ref_pairs (List[int]): sparse vector representation of Orig-Ref pairs
            sparse_pred_pairs (List[int]): sparse vector representation of Orig-Pred pairs

        Returns:
            float: precision of picking the right token
        """
        true_positive_ids = np.multiply(sparse_ref_pairs, sparse_pred_pairs)
        true_positive = np.sum([ref_pairs[idx][1] == pred_pairs[idx][1] for idx in true_positive_ids if idx == 1])
        precision = true_positive / np.sum(true_positive_ids)
        return precision


if __name__ == "__main__":
    # Test
    orig = ["Th cat si on the fride,", "Sumer is commig!"]
    ref = ["The cat is on the fridge.", "Summer is comming!"]
    pred = ["Th big cat is in the fridge.", "Summer is hot!"]

    spellcheck_evaluator = SpellcheckEvaluator(originals=orig, references=ref)
    results = spellcheck_evaluator.evaluate(predictions=pred)
    print(results)
