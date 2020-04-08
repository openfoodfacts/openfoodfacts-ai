from pathlib import Path
from regex import format_percentages
from flashtext import KeywordProcessor

from models.base import BaseModel

# TODO : shall be a paraameter
patterns_path = Path(__file__).parent / 'patterns_fr.txt'


class RegexModel(BaseModel):
    def __init__(self, only_option=None):
        super(RegexModel, self).__init__()
        self.only_option = only_option
        self.keyword_processor = self._load_keywords()

    def _load_keywords(self):
        keyword_processor = KeywordProcessor(case_sensitive=True)
        with patterns_path.open() as f:
            current_replacement = None
            for line in f.readlines():
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    current_replacement = None
                elif current_replacement is None:
                    current_replacement = line
                else:
                    keyword_processor.add_keyword(line, current_replacement)
        return keyword_processor

    def predict(self, items):
        return [self.apply(item['original']) for item in items]

    @property
    def name(self):
        if self.only_option is None:
            return 'RegexModel (all)'
        else:
            return f'RegexModel ({self.only_option})'

    def apply(self, txt):
        if self.only_option is None:
            txt = self.apply_percentages(txt)
            txt = self.apply_replacements(txt)
        elif self.only_option == 'percentages':
            txt = self.apply_percentages(txt)
        elif self.only_option == 'replacements':
            txt = self.apply_replacements(txt)
        return txt

    def apply_percentages(self, txt):
        return format_percentages(txt)

    def apply_replacements(self, txt):
        return self.keyword_processor.replace_keywords(txt)
