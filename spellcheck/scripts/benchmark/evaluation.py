import logging
from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Iterable, List, Iterator

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

    It is designed to evaluate how well Spellcheck models performs on recognizing errors in the
    list of ingredients and correctly fix it in respect of the benchmark.

    Therefore, 3 elements are necessary:
        * Original lists of ingredients as found in Open Food Facts
        * The expected corrections used to define which elements were supposed to be corrected and the ones that were not.
        * The predictions from the model.

    The particularity of this evaluation algorithm comes from calculating the precision, recall, and f1 relative to the errors to correct only, and
    directly extracted from texts.

    To achieve this, texts are tokenized using a Byte Pair Encoding (BPE) tokenizer from the `tiktoken` library.

    The process is then 2-fold:
        * Firstly, encoded originals and references are aligned using Sequence Alignment technique, particularly used in DNA sequences, to locate which tokens
        were transformed, added, or deleted.
        * Secondly, the same process is applied to encoded originals and predictions. Precision, Recall, and F1-score are calculated relatively to the
        errors to fix by comparison with original-references pairs.

    Args:
        originals (List[str]): Batch of original ingredient lists to correct
        references (List[str]): Batch of expected ingredients lists after correction
        encoding_name (str, optional): BPE tokenizer from the tiktoken library. Defaults to "cl100k_base".
    """

    def __init__(
        self,
        originals: List[str],
        references: List[str],
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.originals = originals
        self.references = references
        self.encoder = tiktoken.get_encoding(encoding_name=encoding_name)

    def evaluate(self, predictions: List[str], beta: float = 1.0) -> Mapping:
        """Evaluate the Precision, Recall, F1-score, and F1-beta_score relative the errors to correct.
        For example: "does this error were supposed to be corrected?"

        Note: Precision relative to the right correction not implemented yet.  

        Args:
            predictions (List[str]): _description_
            beta (float, optional): _description_. Defaults to 1.0.

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
            sparsed_ref_pairs = self.convert_pairs_into_sparse(aligned_ref_pairs)
            sparsed_pred_pairs = self.convert_pairs_into_sparse(aligned_pred_pairs)
            assert len(sparsed_ref_pairs) == len(sparsed_pred_pairs), "Ref and pred pairs don't have the same length!"

            inverse_sparsed_ref_pairs = [1 if i == 0 else 0 for i in sparsed_ref_pairs]
            inverse_sparsed_pred_pairs = [1 if i == 0 else 0 for i in sparsed_pred_pairs]

            # Calculate metrics
            true_positive = np.sum(np.multiply(sparsed_ref_pairs, sparsed_pred_pairs))
            false_postive = np.sum(np.multiply(inverse_sparsed_ref_pairs, sparsed_pred_pairs))
            false_negative = np.sum(np.multiply(sparsed_ref_pairs, inverse_sparsed_pred_pairs))

            precision = true_positive / (true_positive + false_postive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            f1_betas.append(f1_beta)

            # Also calculate if model token prediction are correct
            correction_precision = self.compute_correction_precision(
                ref_pairs=aligned_ref_pairs,
                pred_pairs=aligned_pred_pairs,
                sparsed_ref_pairs=sparsed_ref_pairs, 
                sparsed_pred_pairs=sparsed_pred_pairs
            )
            correction_precisions.append(correction_precision)

        # Mean results for the entire batch 
        results = {
            "correction_precision": np.mean(correction_precisions),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
            "f1_beta": np.mean(f1_betas),
            "beta": beta,
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

        Example:    
        ```
        tokens_1 = [791, 8415, 4502, 389, 279, 282, 1425, 11]
        tokens_2 = [791, 8415, 374, 389, 279, 38681, 13]

        alignment = [(791, 791), (8415, 8415), (4502, 374), (389, 389), (279, 279), (282, None), (1425, 38681), (11, 13)]
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
        """Convert Alignement tuples into a sparse vector. 
        If there is a mismatch between tokens from the same pair, it is considered as a modification (=1). 
        
        Args:
            pairs (List[Tuple]): Iterable of token pairs from the Sequence algnment method.
        """
        return [0 if i == j else 1 for i, j in pairs]
    
    @staticmethod
    def align_ref_pred_pairs(
        ref_pairs: List[Tuple], 
        pred_pairs: List[Tuple],
        insert_pairs: Tuple = (None, None)
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """_summary_

        Args:
            ref_pairs (List[Tuple]): _description_
            pred_pairs (List[Tuple]): _description_
            insert_pairs (Tuple, optional): _description_. Defaults to (None, None).

        Returns:
            Tuple[List[Tuple], List[Tuple]]: _description_
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
        sparsed_ref_pairs: List[int], 
        sparsed_pred_pairs: List[int]
    ) -> float:
        """"""
        corrected_tokens_ids = np.multiply(sparsed_ref_pairs, sparsed_pred_pairs)
        true_positive = np.sum([ref_pairs[idx][1] == pred_pairs[idx][1] for idx in corrected_tokens_ids if idx == 1])
        precision = true_positive / np.sum(corrected_tokens_ids)
        return precision


if __name__ == "__main__":
    # Test
    orig = ["The cat si on the fride,", "Sumer is commig!"]
    ref = ["The cat is on the fridge.", "Summer is comming!"]
    pred = ["The big cat is in the fridge.", "Summer is hot!"]

    # # Example usage
    # list1 = [932, 582, 1494, 14127, 288, 62, 390, 59801, 2445, 5169, 311, 267, 2172, 13, 721, 51, 8875, 300, 409, 1448, 21067, 409, 59801, 2445, 5169, 5056, 1000]
    # list2 = [932, 582, 1494, 14127, 288, 62, 390, 59801, 2445, 5169, 311, 267, 2172, 13, 350, 8875, 300, 409, 1448, 21067, 409, 59801, 2445, 5169, 13]
    spellcheck_evaluator = SpellcheckEvaluator(originals=orig, references=ref)
    results = spellcheck_evaluator.evaluate(predictions=pred)
    print(results)
