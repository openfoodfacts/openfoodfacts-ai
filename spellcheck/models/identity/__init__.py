from models.base import BaseModel


class IdentityModel(BaseModel):
    def predict(self, items):
        return [item['original'] for item in items]
