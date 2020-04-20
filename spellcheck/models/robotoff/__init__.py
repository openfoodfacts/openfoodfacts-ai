import json
import requests
from models.base import BaseModel


class RobotoffAPIModel(BaseModel):
    URL = "https://robotoff.openfoodfacts.org/api/v1/predict/ingredients/spellcheck"

    def __init__(self, index="product", confidence=1):
        super(RobotoffAPIModel, self).__init__()
        self.index = index
        self.confidence = confidence

    @property
    def name(self):
        return f"RobotoffAPI__index_{self.index}__conf_{self.confidence}"

    def apply_one(self, txt):
        try:
            r = requests.get(self.URL, params=self.get_params(text=txt))
            data = r.json()
            if data["corrected"] != "":
                return data["corrected"]
            else:
                return data["text"]
        except Exception as e:
            return "FAILURE"

    def get_params(self, **kwargs):
        return {"index": self.index, "confidence": self.confidence, **kwargs}
