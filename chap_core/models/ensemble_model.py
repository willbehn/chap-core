import numpy as np

from chap_core.datatypes import Samples
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class EnsembleModel(ConfiguredModel):
    def __init__(self, models: list[ConfiguredModel]):
        self._models = models

    def train(self, train_data: DataSet, extra_args=None):
        for model in self._models:
            model.train(train_data, extra_args)
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        print("\n\nEnsemble prediction:")

        predictions = [model.predict(historic_data, future_data) for model in self._models]

        for i,pred in enumerate(predictions):
            print(f"Modell {i+1} prediction:\n {pred}");

        averaged = {}

        print(f"Locations i predictions: {predictions[0].keys()}")

        for loc in predictions[0].keys():
            avg_samples = np.mean([pred[loc].samples for pred in predictions], axis=0)

            averaged[loc] = Samples(predictions[0][loc].time_period, avg_samples)

        return DataSet(averaged)
