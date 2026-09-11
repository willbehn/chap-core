import logging
import time

logger = logging.getLogger(__name__)

from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chap_core.explainability.lime import (
    prepare_explain_inputs,
    build_feature_map
)

"""
TODO fjern etterhvert
Foreløpig plan: 
Dele data pipeline med LIME: prepare_explain_inputs -> feature mapping
Ny shap del: Skrive value func som tar mask og oversetter til modell kall
    perturb_vectors -> produce_lime dataset? 
Sende dette inn i KernelExplainer med 0 som background, og 1 som på
"""

# Some of the samplers for lime might not work as intended for shap, will update
# with more samplers
def _check_allowed_sampler(sampler_name: str):
    allowed_samplers = {"global_mean", "background", "random"}

    assert sampler_name in allowed_samplers, (
        "Sampler not supported"
    )

    return sampler_name


def explain_shap(
        model: ExternalModel,
        dataset: DataSet,
        location: str,
        horizon: int,
        granularity: int = 10,
        num_perturbations: int = 300,
        segmenter_name: str = "uniform",
        sampler_name: str = "global_mean",
        last_n: int | None = None,
        seed: int | None = None,
        timed: bool = False,
    ):
    
    start = time.perf_counter()
    if timed:
        logger.info("Started SHAP pipeline")

    
    inputs = prepare_explain_inputs(
            dataset=dataset,
            location=location,
            horizon=horizon,
            segmenter_name=segmenter_name,
            granularity=granularity,
            sampler_name=_check_allowed_sampler(sampler_name),
            seed=seed,
            last_n=last_n,
            timed=timed,
            start=start,
        )

    print(f"inputs AFTER prepare_explain_inputs:\n{inputs}\n")

    feature_map = build_feature_map(inputs.x0)

    print(f"feature_map AFTER build_feature_map:\n{feature_map}\n")
  