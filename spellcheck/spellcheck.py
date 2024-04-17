"""Spellcheck module."""
import logging

from utils.model import BaseModel


LOGGER = logging.getLogger(__name__)


class SpellChecker:
    """Spellcheck module to correct typos and errors in lists of ingredients.
    
    Args:
        model (BaseModel): model used for correcting list of ingredients.
    """
    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def predict(self, list_of_ingredients: str) -> str:
        """Correct list of ingredients
        
        Args:
            list_of_ingredients (str): List of ingredients to correct.

        Return:
            (str): Corrected list of ingredients.
        """
        correction = self.model.predict(text=list_of_ingredients)
        return correction
