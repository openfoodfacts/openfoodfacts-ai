"""
/!\ /!\ /!\
Under development repo.
Code copy-pasted from :
https://github.com/openfoodfacts/robotoff/blob/4edbc715d81e84f234cc284222697632cf5b13ee/robotoff/ingredients.py
/!\ /!\ /!\

"""

from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple, Dict

from spacy.lang.fr import French

SPLITTER_CHAR = {"(", ")", ",", ";", "[", "]", "-", "{", "}"}

# Food additives (EXXX) may be mistaken from one another, because of their edit distance proximity

OffsetType = Tuple[int, int]


FR_NLP = French()


class TokenLengthMismatchException(Exception):
    pass


def normalize_ingredients(ingredients: str) -> str:
    normalized = ingredients.lower()
    normalized = normalized.replace("œu", "oeu")
    normalized = normalized.replace("’", "'")
    return normalized


def normalize_item_ingredients(item: Dict) -> Dict:
    item = item.copy()
    keys = ["original", "correct", "prediction"]
    for key in keys:
        if key in item:
            item[key] = normalize_ingredients(item[key])
    return item


def tokenize_ingredients(text: str) -> List[str]:
    tokens = []
    for token in FR_NLP(text):
        tokens.append(token.orth_)

    tokens = [token for token in tokens if any(c.isalnum() for c in token)]
    tokens = [token.rstrip("s") for token in tokens]
    return tokens


def format_ingredients(ingredients_txt: str) -> Set[str]:
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
