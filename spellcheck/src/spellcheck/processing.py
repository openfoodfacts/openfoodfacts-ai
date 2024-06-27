from typing import Iterable
import re

from spellcheck.utils import get_logger


LOGGER = get_logger()


class DataProcessor:
    """Class methods to process datasets for the Spellcheck."""

    @classmethod
    def align_oe(
        cls,
        references: Iterable[str], 
        texts: Iterable[str]
    ) -> Iterable[str]:
        """Align "oe" - "œ character between the reference text and the target.

        Examples:
        ref = "bœuf, œuf, ..."
        text = "boeuf, œuf, ..."
        output = "bœuf, œuf, ..."

        Args:
            references (Iterable[str]): Text references.
            text (Iterable[str]): Texts to align with references.
        """
        return [
            cls._align_oe(ref, text)
            for ref, text in zip(references, texts)
        ]
    
    def _align_oe(reference: str, text: str) -> str:
        """
        Args:
            reference (str): Text references.
            text (str): Texts to align with references.

        Returns:
            str: Text with identical oe.
        """
        pattern = re.compile(r"oe|œ") # OR
        
        ref_matches = pattern.finditer(reference)
        text_matches = pattern.finditer(text)

        # We distinguish Matches in the dictionnary by their unique Span
        matches_dict = {
            text_match.span(): ref_match.group()
            for text_match, ref_match in zip(text_matches, ref_matches)
        }

        def replace_match(match: re.Match) -> str:
            """Sub-function for the re.sub algorithm.
            For each match found, it compares oe in both the reference and text and change oe in text if different

            Args:
                match (re.Match): Match in text

            Returns:
                str: Replaced match
            """
            ref_match = matches_dict.get(match.span())
            # Match.group() return the match as a string
            # Possibility a oe was not found in the reference, leading to dict.get()= None
            if ref_match and ref_match != match.group(): 
                return ref_match # Reference
            else:
                return match.group()
        
        modified_text = pattern.sub(replace_match, text)
        return modified_text

    @classmethod
    def align_whitespace_percentage(
        cls,
        references: Iterable[str],
        texts: Iterable[str],
    ) -> Iterable[str]:
        """ "Matches the whitespace between the number and the percentage for each percentage in the list of ingredients.

        Example:
        ref = "patata 15%, piment 15,2%, pamplemousse 47 %, aubergine 18.1 "
        text = "patata 15 %, piment 15,2 %, pamplemousse 47 %, aubergine 18.1%"
        output = "patata 15%, piment 15,2%, pamplemousse 47 %, aubergine 18.1 %"

        Args:
            references (Iterable[str]): Text references.
            texts (Iterable[str]): Texts to align with references.

        Returns:
            Iterable[str]: Texts with aligned whitespaces.
        """
        return [
            cls._align_whitespace_percentage(ref, text)
            for ref, text in zip(references, texts)
        ]

    def _align_whitespace_percentage(
        reference: str,
        text: str,
    ) -> str:
        """
        Args:
            reference (str): Text references.
            text (str): Texts to align with references.

        Returns:
            str: Text with aligned whitespaces.
        """
        # 3 groups: "15,5 %"" => "15,5" - " " - "%"
        # Can also be 14'% => "14" - "'" - "%"
        pattern = re.compile(r"([,\.\d]*)(\D{0,2})(%)")

        # Create a dictionnary with text match and the reference gap between
        ref_matches = pattern.finditer(reference)
        text_matches = pattern.finditer(text)
        # We distinguish Matches in the dictionnary by their unique Span
        matches_dict = {
            text_match.span(): ref_match.group(2)
            for text_match, ref_match in zip(text_matches, ref_matches)
        }

        def replace_match(match: re.Match) -> str:
            """Sub-function for the re.sub algorithm.
            For each match found, it looks up into the prepared dictionnary {re.Match: "str between number and % in ref"}

            Args:
                match (re.Match): Match in text

            Returns:
                str: Replaced match
            """
            # Compose the new match with the space between group 1 and group 3
            text_whitespace = match.group(2)
            ref_whitespace = matches_dict.get(match.span())
            if ref_whitespace in ["", " "] and text_whitespace in ["", " "]:  # Could be empty string "" or whitespace " "
                return match.group(1) + ref_whitespace + match.group(3)
            else:
                LOGGER.debug(
                    f"Match returned unchanged: match in text to align={match}; reference={reference}; text to align based on reference={text}"
                )
                return match.group(0)

        modified_text = pattern.sub(replace_match, text)
        return modified_text
