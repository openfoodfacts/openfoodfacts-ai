from pathlib import Path
from models.regex.utils import format_percentages

from vocabulary import Vocabulary
from ingredients import tokenize_ingredients
from models.base import BaseModel

# TODO : shall be a paraameter
patterns_path = Path(__file__).parent / "patterns_fr.txt"


class RegexModel(BaseModel):
    def __init__(self, mode=None):
        super(RegexModel, self).__init__()
        self.mode = mode
        if mode is None or mode == "replacements":
            self.replacements = self._load_replacements()
        if mode is None or mode == "vocabulary":
            self.wikipedia_voc = Vocabulary("wikipedia_lower")
            self.ingredients_voc = Vocabulary("ingredients_fr_tokens") | Vocabulary(
                "ingredients_fr"
            )

    def _load_replacements(self):
        replacements = {}
        with patterns_path.open() as f:
            current_replacement = None
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#") or len(line) == 0:
                    current_replacement = None
                elif current_replacement is None:
                    current_replacement = line
                else:
                    replacements[line.lower()] = current_replacement.lower()
                    replacements[line.upper()] = current_replacement.upper()
                    replacements[line.capitalize()] = current_replacement.capitalize()
                    replacements[line] = current_replacement
        return replacements

    @property
    def name(self):
        if self.mode is None:
            return "RegexModel__all"
        else:
            return f"RegexModel__{self.mode}"

    def apply_one(self, txt):
        methods = {
            "percentages": self.apply_percentages,
            "vocabulary": self.apply_vocabulary,
            "replacements": self.apply_replacements,
            "punctuation": self.apply_punctuation,
        }
        if self.mode is None:
            for method, apply_function in methods.items():
                txt = apply_function(txt)
        else:
            txt = methods[self.mode](txt)
        return txt

    def apply_percentages(self, txt: str) -> str:
        return format_percentages(txt)

    def apply_replacements(self, txt: str) -> str:
        for key, value in self.replacements.items():
            txt = txt.replace(key, value)
        return txt

    def apply_vocabulary(self, txt: str) -> str:
        for token in tokenize_ingredients(txt, remove_additives=True):
            if all(c.isalpha() for c in token):
                if not token in self.wikipedia_voc:
                    suggestion = self.ingredients_voc.suggest_one(token)
                    if suggestion is not None:
                        txt = txt.replace(token, suggestion)
        return txt

    def apply_punctuation(self, txt: str) -> str:
        txt = " ".join(txt.split())
        return txt
