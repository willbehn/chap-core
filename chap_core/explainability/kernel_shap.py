import logging 

logger = logging.getLogger(__name__)

from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chap_core.explainability.lime import (
    prepare_explain_inputs
   
)
"""
Foreløpig plan: 
Dele data pipeline med LIME: prepare_explain_inputs -> feature mapping
Ny shap del: Skrive value func som tar mask og oversetter til modell kall
    perturb_vectors -> produce_lime dataset? 
Sende dette inn i KernelExplainer med 0 som background, og 1 som på
"""


def explain_shap(
        model: ExternalModel,
        dataset: DataSet,
        location: str,
        horizon: int,
        granularity: int = 10,
        num_perturbations: int = 300,
        segmenter_name: str = "uniform",
        sampler_name: str = "background",
        last_n: int | None = None,
        seed: int | None = None,
        timed: bool = False,
):
    print(test)

    start = time.perf_counter()
    if timed:
        logger.info("Started LIME pipeline")
    
    inputs = prepare_explain_inputs(
            dataset=dataset,
            location=location,
            horizon=horizon,
            segmenter_name=segmenter_name,
            granularity=granularity,
            sampler_name=sampler_name,
            seed=seed,
            last_n=last_n,
            timed=timed,
            start=start,
        )