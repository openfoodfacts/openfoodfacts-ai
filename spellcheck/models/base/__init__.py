from utils import save_dataset
from paths import new_experiment_path


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.last_experiment_path = None

    def predict(self, items):
        return [self.apply_one(item['original']) for item in items]

    def apply_one(self, txt):
        raise NotImplementedError

    def predict_save(self, items):
        self.last_experiment_path = new_experiment_path(self.name)
        predictions = self.predict(items)

        for item, prediction in zip(items, predictions):
            item['prediction'] = prediction

        save_dataset(self.last_experiment_path, items)

        for item, prediction in zip(items, predictions):
            del item['prediction']

        return predictions

    @property
    def name(self):
        return self.__class__.__name__
