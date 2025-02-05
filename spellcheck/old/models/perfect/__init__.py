from models.base import BaseModel


class PerfectModel(BaseModel):
    def predict(self, items):
        return [item["correct"] for item in items]
