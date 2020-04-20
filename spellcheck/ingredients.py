"""
/!\ /!\ /!\
Under development repo.
Code copy-pasted from :
https://github.com/openfoodfacts/robotoff/blob/4edbc715d81e84f234cc284222697632cf5b13ee/robotoff/ingredients.py

# TODO: Find proper way to import `process_ingredients`
/!\ /!\ /!\

"""

from dataclasses import dataclass, field
from typing import List, Tuple, Iterable


SPLITTER_CHAR = {"(", ")", ",", ";", "[", "]", "-", "{", "}"}

# Food additives (EXXX) may be mistaken from one another, because of their edit distance proximity

OffsetType = Tuple[int, int]


class TokenLengthMismatchException(Exception):
    pass


def format_ingredients(ingredients_txt):
    ingredients = {
        " ".join(ingredient.split())
        for ingredient in process_ingredients(
            ingredients_txt
        ).iter_normalized_ingredients()
    }
    return {ingredient for ingredient in ingredients if len(ingredient) > 0}


@dataclass
class Ingredients:
    text: str
    normalized: str
    offsets: List[OffsetType] = field(default_factory=list)

    def iter_normalized_ingredients(self) -> Iterable[str]:
        for start, end in self.offsets:
            yield self.normalized[start:end]

    def get_ingredient(self, index) -> str:
        start, end = self.offsets[index]
        return self.text[start:end]

    def get_normalized_ingredient(self, index) -> str:
        start, end = self.offsets[index]
        return self.normalized[start:end]

    def ingredient_count(self) -> int:
        return len(self.offsets)


def normalize_ingredients(ingredients: str) -> str:
    normalized = ingredients.lower()
    return normalized


def process_ingredients(ingredient_text: str) -> Ingredients:
    offsets = []
    chars = []

    normalized = normalize_ingredients(ingredient_text)
    start_idx = 0

    for idx, char in enumerate(normalized):
        if char in SPLITTER_CHAR:
            offsets.append((start_idx, idx))
            start_idx = idx + 1
            chars.append(" ")
        else:
            chars.append(char)

    if start_idx != len(normalized):
        offsets.append((start_idx, len(normalized)))

    normalized = "".join(chars)
    return Ingredients(ingredient_text, normalized, offsets)
