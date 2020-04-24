from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Optional, AbstractSet

file_path = Path(__file__)


class Vocabulary(object):

    FILENAMES = {
        "wikipedia_lower": "strings_lower.txt",
        "ingredients_fr": "ingredients_fr.txt",
        "ingredients_fr_tokens": "ingredients_fr_tokens.txt",
    }

    def __init__(
        self, voc_name: Optional[str] = None, tokens: Optional[AbstractSet] = None
    ):
        self.voc = set()
        if voc_name is not None:
            try:
                path = file_path.parent / self.FILENAMES[voc_name]
                with path.open("r") as f:
                    self.voc = set(
                        self.normalize(token) for token in f.read().split("\n")
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Repo is currently under development. Vocabulary files have not been committed yet."
                )

        if tokens is not None:
            self.voc.update(tokens)

        self.deaccented_tokens = defaultdict(list)
        for token in self.voc:
            deaccented_token = self.deaccent(token)
            if deaccented_token != token:
                self.deaccented_tokens[deaccented_token].append(token)

    def __contains__(self, token: str) -> bool:
        return self.normalize(token) in self.voc

    def __or__(self, other):
        return Vocabulary(tokens=self.voc | other.voc)

    def suggest_one(self, token: str) -> Optional[str]:
        suggestions = self.suggest_deaccent(token)
        if suggestions is not None:
            if len(suggestions) == 1:
                return suggestions[0]
            elif len(suggestions) > 1:
                return

        suggestions = self.suggest_split(token)
        if len(suggestions) == 1:
            suggestion = suggestions[0]
            if suggestion[1] != "s":  # Keep plural forms
                if suggestion[1] != "e":  # Keep last "e"
                    return suggestion[0] + " " + suggestion[1]

    def suggest_deaccent(self, token: str) -> Optional[str]:
        if token in self:
            return
        deaccented_token = self.deaccent(token)
        if deaccented_token in self.deaccented_tokens:
            return self.deaccented_tokens[deaccented_token]

    def suggest_split(self, token: str) -> List[Tuple[str, str]]:
        if token in self:
            return []

        suggestions = []
        for i in range(len(token)):
            pre = token[:i]
            post = token[i:]
            if pre in self and post in self or (pre + " " + post) in self:
                suggestions.append((pre, post))
        return suggestions

    @staticmethod
    def deaccent(token: str) -> str:
        ACCENTS = {"a": "à", "e": "éêè", "u": "ùüû", "i": "ïî"}
        for letter in ACCENTS:
            for c in ACCENTS[letter]:
                token = token.replace(c, letter)
        return token

    @staticmethod
    def normalize(token: str) -> str:
        return token.lower()
