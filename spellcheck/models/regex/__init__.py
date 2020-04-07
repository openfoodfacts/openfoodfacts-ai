from regex import format_percentages

from models.base import BaseModel


class RegexModel(BaseModel):
    REPLACE_LIST = {
        '&quot;': '*',
        'oeu': 'œu',
        'OEU': 'ŒU',
        'OEu': 'Œu',
        'Oeu': 'Œu',
    }

    def __init__(self, only_option=None):
        super(RegexModel, self).__init__()
        self.only_option = only_option

    def predict(self, items):
        return [self.apply_regex(item['original']) for item in items]

    @property
    def name(self):
        if self.only_option is None:
            return 'RegexModel (all)'
        else:
            return f'RegexModel ({self.only_option})'

    def apply_regex(self, txt):
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
        for replace_key, replace_value in self.REPLACE_LIST.items():
            txt = txt.replace(replace_key, replace_value)
        return txt
