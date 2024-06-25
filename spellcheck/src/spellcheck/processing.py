from typing import Iterable
import re

from spellcheck.utils import get_logger


LOGGER = get_logger()


class DataProcessor:
    """Class methods to process datasets for the Spellcheck."""

    @staticmethod
    def uniformize_oe(references: Iterable[str], texts: Iterable[str]):
        """Align "oe" - "œ character between the reference text and the target.

        Examples:
        ref = "bœuf, œuf, ..."
        text = "boeuf, œuf, ..."
        output = "bœuf, œuf, ..."

        Args:
            references (Iterable[str]): Text references.
            text (Iterable[str]): Texts to align with references.
        """
        for reference, text in zip(references, texts):
            if "oe" in reference:
                text = text.replace("œ", "oe")
            elif "œ" in reference:
                text = text.replace("oe", "œ")
            yield text

    @classmethod
    def uniformize_whitespace_percentage(
        cls,
        references: Iterable[str],
        texts: Iterable[str],
    ) -> Iterable[str]:
        """ "Matches the whitespace between the number and the percentage for each percentage in the list of ingredients.

        Example:
        ref = "patata 15%, piment 15,2%, pamplemousse 47_%, aubergine 18.1_%"
        text = "patata 15_%, piment 15,2_%, pamplemousse 47_%, aubergine 18.1%"
        output = "patata 15%, piment 15,2%, pamplemousse 47_%, aubergine 18.1_%"

        Args:
            references (Iterable[str]): Text references.
            texts (Iterable[str]): Texts to align with references.

        Returns:
            Iterable[str]: Texts with aligned whitespaces.
        """
        return [
            cls.__uniformize_whitespace_percentage(ref, text)
            for ref, text in zip(references, texts)
        ]

    def __uniformize_whitespace_percentage(
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
        pattern = re.compile(r"(\d+[,\.\d]*)(\s*)(%)")

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
            whitespace = matches_dict.get(match.span())
            LOGGER.info("hello")
            if whitespace in ["", " "]:  # Could be empty string "" or whitespace " "
                return match.group(1) + whitespace + match.group(3)
            else:
                LOGGER.warning(
                    f"Match not found in Matches dictionnary: Dictionnary {matches_dict}; Match: {match.group(0)}; Match span: {match.span()}; text: {text}; reference: {reference}. Match is returned unchanged."
                )

        modified_text = pattern.sub(replace_match, text)
        return modified_text
