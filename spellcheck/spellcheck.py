from utils.model import BaseModel

class SpellChecker:
    """"""
    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def predict(self, list_of_ingredients: str) -> str:
        """"""
        correction = self.model.predict(text=list_of_ingredients)
        return correction
