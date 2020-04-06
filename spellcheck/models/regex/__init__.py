from regex import format_percentages

from models.base import BaseModel


class RegexModel(BaseModel):
    def predict(self, items):
        return [self.apply_regex(item['original']) for item in items]

    def apply_regex(self, txt):
        txt = format_percentages(txt)
        return txt
