from regex import format_percentages

from models.base import BaseModel


class RegexModel(BaseModel):
    def predict(self, items):
        return [self.apply_regex(item['original']) for item in items]

    def apply_regex(self, txt):
        txt = format_percentages(txt)
        for replace_key, replace_value in REPLACE_LIST.items():
            txt = txt.replace(replace_key, replace_value)
        return txt


REPLACE_LIST = {
    '&quot;': '*',
    'oeu': 'œu',
    'OEU': 'ŒU',
    'OEu': 'Œu',
    'Oeu': 'Œu',
}
