from utils import save_dataset
from paths import new_experiment_path


class BaseModel(object):
    def predict(self, items):
        raise NotImplementedError

    def predict_save(self, items):
        path = new_experiment_path(self.__class__.__name__)
        predictions = self.predict(items)

        for item, prediction in zip(items, predictions):
            item['prediction'] = prediction

        save_dataset(path, items)

        for item, prediction in zip(items, predictions):
            del item['prediction']

        return predictions
