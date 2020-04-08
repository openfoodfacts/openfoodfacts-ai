import json
import requests
from models.base import BaseModel


class RobotoffAPIModel(BaseModel):
    URL = 'https://robotoff.openfoodfacts.org/api/v1/predict/ingredients/spellcheck'

    def apply_one(self, txt):
        try:
            r = requests.get(self.URL, params={'text': txt})
            data = r.json()
            if data['corrected'] != '':
                return data['corrected']
            else:
                return data['text']
        except Exception as e:
            return 'FAILURE'
