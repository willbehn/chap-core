import logging
import time

import numpy as np
import shap

from chap_core.explainability.lime import (
    ExplainInputs,
    build_feature_map,
    perturb_vectors,
    predict_pertubations,
    prepare_explain_inputs,
)
from chap_core.explainability.segment import SplitSegmentation
from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)

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
    allowed_samplers = {"global_mean", "background"} # TODO expand with other samplers later

    assert sampler_name in allowed_samplers, (
        "Sampler not supported"
    )

    return sampler_name


def _explain_shap(
    *,
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    num_perturbations: int,
    seed: int | None,
    inputs: ExplainInputs,
) -> list[tuple[str, float]]:

    #print(f"inputs AFTER prepare_explain_inputs:\n{inputs}\n")

    feature_map = build_feature_map(inputs.x0)
    num_features = len(feature_map)
    feature_names = [name for name, _, _ in feature_map]

    #print(f"\n\nNUMBER OF FEATURES: {num_features}\n\n")
    #print(f"feature_map AFTER build_feature_map:\n{feature_map}\n")

    def value_fn(masks: np.ndarray) -> np.ndarray:
        #print(masks)

        perturbations, perturbation_masks = perturb_vectors(
            inputs.hist_df,
            inputs.x0,
            inputs.feat_indices,
            inputs.sampler,
            feature_map,
            masks,
            inputs.global_means
        )

        # TODO currently explains the mean, and last timestep, todo for later
        _, y, _, _ = predict_pertubations(
            model,
            inputs.hist_df,
            inputs.future_df,
            perturbations,
            perturbation_masks,
            feature_names,
            inputs.features_hist,
            inputs.features_fut,
            horizon,
            location,
            inputs.feat_indices,
            inputs.hist_type,
            inputs.fut_type,
            dataset,
            inputs.full_future_weather,
        )

        return np.array(y)


    np.random.seed(seed=seed)
    background = np.zeros((1,num_features))
    instance = np.ones((1, num_features))

    explainer = shap.KernelExplainer(value_fn, background)

    shap_values = explainer.shap_values(instance, nsamples=num_perturbations, l1_reg=False)

    results = sorted(
        zip(feature_names, np.asarray(shap_values).ravel().tolist(), strict=True),
        key=lambda item: item[0]
    )
    return results


def explain(
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int = 3,
    granularity: int = 10,
    num_perturbations: int = 300,
    segmenter_name: str = "uniform",
    sampler_name: str = "global_mean",
    last_n: int | None = None,
    seed: int | None = None,
    timed: bool = False,
) -> list[tuple[str, float]]:

    start = time.perf_counter()
    if timed:
        logger.info("Started SHAP pipeline")

    """
    psudokode:
    if (scope_explanation):
        dataset, last_n = prune_temporal_depth(model, dataset, location)
    """

    inputs = prepare_explain_inputs(
        dataset=dataset,
        location=location,
        horizon=horizon,
        segmenter=segmenter_name,
        granularity=granularity,
        sampler_name=_check_allowed_sampler(sampler_name),
        seed=seed,
        last_n=last_n,
        timed=timed,
        start=start,
    )

    result = _explain_shap(
        model=model,
        dataset=dataset,
        location=location,
        horizon=horizon,
        num_perturbations=num_perturbations,
        seed=seed,
        inputs=inputs,
    )

    end = time.perf_counter()
    if timed:
        logger.info(f"SHAP pipeline done, used {end-start}")

    return result

# Pruning strategy based on TimeSHAP, prunes the depth of the dataset
def prune_temporal_depth(
    *,
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    granularity: int,
    num_perturbations: int,
    sampler_name: str,
    last_n: int | None,
    seed: int | None,
    timed: bool,
    start: float,
    threshold: float):

    """
    psudokode:
    for i in {max_depth-1, max_depth-2 osv til}:
        koalisjoner = lag_koalisjoner(i), idk her ennå

        splitte datasett inn i to grupper, en med gammel, og en med ny, basert på i
        Regne ut shap verdier for hver gruppe med kernel shap

        Hvis shap verdi for gammelgruppe < threshold
            return i, der i er tidssteget man pruner på 

    retun 0, ingen pruning lenger
    """

    max_depth = len(dataset.period_range)
    print(f"STARTING TEMPORAL PRUNING, max depth={max_depth}")

    # Ranges fdrom max_depth -1 to 1
    for split_index in range(max_depth - 1, 0, -1):

        """
        TODO regarding segmentation REMEBER TO REMOVE
        I need to pass down arguemtns to the segmenter from this call since I will need to call it multiple times.
        My solution now is to change segmetner to also accept a segmentation model.
        Issues: Comlpicated code, and I need to rerun data preperation for each time I want to change the segmentation
        Solution: I will refactor it later to first do data prerperation -> call segment() that does the segmentation. 
        """
        inputs = prepare_explain_inputs(
            dataset=dataset,
            location=location,
            horizon=horizon,
            segmenter=SplitSegmentation(split_index),
            granularity=granularity,
            sampler_name=_check_allowed_sampler(sampler_name),
            seed=seed,
            last_n=last_n,
            timed=timed,
            start=start
        )

        result = _explain_shap(
            model=model,
            dataset=dataset,
            location=location,
            horizon=horizon,
            num_perturbations=num_perturbations,
            seed=seed,
            inputs=inputs,
        )

        # TODO uses only the first feature as of now
        old_seg = next((seg, sv) for seg, sv in result if "_seg_1" in seg)
        new_seg = next((seg, sv) for seg, sv in result if "_seg_0" in seg)

        print(f"OLD SEGMENT for split_index {split_index}: {old_seg}")

        old_share = abs(old_seg[1])/ (abs(old_seg[1]) + abs(new_seg[1]))
        print(f"SHARE: {old_share}")

        if old_share < threshold:
            return split_index

    return 0
        
