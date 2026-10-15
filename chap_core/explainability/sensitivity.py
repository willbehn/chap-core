import logging

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


"""
TODO fjern
Plan for første iterasjon av sensitivy analysis av chap modeller:
- Main funksjon so tar inn datasett, model, evt andre parametere som context legngden 
    - Her er tanken å følge lime og kernel shap sitt interface, da ta inn ferdig trente modeller, og heller
    utlede splits osv selv
- Egen Funksjon som regner ut baseline, hva den predikerer i den perioden med dataen den er gitt
    - Her tenker jeg at jeg også evt kan se om jeg kan bruke feks backtest/evalute sin predikjson så jeg slipper
    å gjøre det på nytt?
- Deretter ha en generisk metode som støtter at man kjører prediksjoner over ulike datasett, feks 
mean av tidligere år, maks a tidligere år, min av tidligere år osv. 
- TODO til senere: finne ut av hvordan jeg skal evualuere forskjellen mellom runs
"""

def sensitivity_analysis(
        model: ExternalModel, 
        dataset: DataSet,
        num_unused_periods: int,
        prediction_length: int = 3,
        stride: int = 1,
    ):

    #TODO the model must have been trained on only data before the last num_unused_periods,
    #need to look into how to verify/check this
    n_origins = (num_unused_periods - prediction_length) // stride + 1

    #TODO add checks for bad inputs

    _, splits = train_test_generator(
        dataset,
        prediction_length=prediction_length,
        n_test_sets=n_origins,
        stride=stride,
        future_weather_provider='observed',
    )

    splits = list(splits)
