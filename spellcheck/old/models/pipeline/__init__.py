from models.base import BaseModel


class PipelineModel(BaseModel):
    def __init__(self, models):
        self.models = models

    def apply_one(self, txt):
        for model in self.models:
            txt = model.apply_one(txt)
        return txt

    @property
    def name(self):
        return "PipelineModel__" + "__".join([model.name for model in self.models])
