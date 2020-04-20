from typing import Dict
import requests
from models.base import BaseModel

from joblib import Memory

memory = Memory("~/.joblib", verbose=0)


URL = "https://robotoff.openfoodfacts.org/api/v1/predict/ingredients/spellcheck"


class RobotoffAPIModel(BaseModel):
    def __init__(self, index="product", confidence=1):
        super(RobotoffAPIModel, self).__init__()
        self.index = index
        self.confidence = confidence

    @property
    def name(self):
        return f"RobotoffAPI__index_{self.index}__conf_{self.confidence}"

    def apply_one(self, text: str) -> str:
        return get_correction(text, self.get_params(text=text))

    def get_params(self, **kwargs):
        return {"index": self.index, "confidence": self.confidence, **kwargs}


@memory.cache
def get_correction(text: str, params: Dict) -> str:
    try:
        r = requests.get(URL, params=params)
        data = r.json()
        if data["corrected"] != "":
            return data["corrected"]
        else:
            return data["text"]
    except Exception:
        return "FAILURE"
