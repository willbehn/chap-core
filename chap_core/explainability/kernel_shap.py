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


# Some of the samplers for lime might not work as intended for shap, will update
# with more samplers
def _check_allowed_sampler(sampler_name: str):
    allowed_samplers = {"global_mean", "background", "fourier"}  # TODO expand with other samplers later

    if sampler_name not in allowed_samplers:
        raise ValueError("Sampler not supported")

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
    feature_map: list[tuple[str, str, int | None]],
    groups: list[int] | None = None,
    player_names: list[str] | None = None,
) -> list[tuple[str, float]]:

    # print(f"inputs AFTER prepare_explain_inputs:\n{inputs}\n")
    feature_names = [name for name, _, _ in feature_map]

    # print(f"\n\nNUMBER OF FEATURES: {num_features}\n\n")
    # print(f"feature_map AFTER build_feature_map:\n{feature_map}\n")

    # TODO make this more robust
    if (groups is None) != (player_names is None):
        raise ValueError("groups and player_names must be passed togheter")

    if groups is None:
        features_to_groups, player_names = list(range(len(feature_map))), feature_names

    features_to_groups = np.asarray(groups)

    #print(f"Feature to groups: {features_to_groups}")

    def value_fn(player_masks: np.ndarray) -> np.ndarray:
        masks = player_masks[:, features_to_groups]
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

    num_players = features_to_groups.max() + 1

    np.random.seed(seed=seed)
    background = np.zeros((1, num_players))
    instance = np.ones((1, num_players))

    explainer = shap.KernelExplainer(value_fn, background)
    shap_values = explainer.shap_values(instance, nsamples=num_perturbations, l1_reg=False)

    results = sorted(zip(player_names, np.asarray(shap_values).ravel().tolist(), strict=True), key=lambda item: item[0])
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
    scope_explanation: bool = False,
    prune_threshold: float | None = None,
) -> list[tuple[str, float]]:

    start = time.perf_counter()
    if timed:
        logger.info("Started SHAP pipeline")

    if scope_explanation:
        if prune_threshold is None:
            raise ValueError("prune_threshold must be set when scope_explanation is True")
        
        split_index = find_temporal_prune_depth(
            model=model,
            dataset=dataset,
            location=location,
            horizon=horizon,
            granularity=granularity,
            num_perturbations=num_perturbations,
            sampler_name=sampler_name,
            last_n=last_n,
            seed=seed,
            timed=timed,
            start=start,
            threshold=prune_threshold,
        )

        if split_index > 0:
            last_n = len(dataset.period_range) - split_index

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

    # TODO refactor prepare_explain_inputs to prepare_data() -> segment_data()
    feature_map = build_feature_map(inputs.x0)

    result = _explain_shap(
        model=model,
        dataset=dataset,
        location=location,
        horizon=horizon,
        num_perturbations=num_perturbations,
        seed=seed,
        inputs=inputs,
        feature_map=feature_map,
    )

    end = time.perf_counter()
    if timed:
        logger.info(f"SHAP pipeline done, used {end - start}")

    return result

# Helper function that removes static features from feature_map. This will result in the static features allways have 
# their actual value when running predict for when both groups are on.
# TODO not sure if I should drop it yet or not
def drop_static_features(
    feature_map: list[tuple[str, str, int | None]]
) -> list[tuple[str, str, int | None]]:

    # Keeps only features that has lags or that is a future feature
    return [entry for entry in feature_map if entry[2] is not None or "_fut_" in entry[0]]


# Helper function to group features into the old/new player game when doing timeshap-ish pruning
def group_two_player_split(feature_map: list[tuple[str, str, int | None]]) -> list[int]:
    #print(feature_map)

    # 0 is new, 1 is old, _fut_ will be added in new segments
    # TODO think also _seg_10... etc can be added, fix later
    groups = [1 if "_seg_1" in seg else 0 for seg, _, _ in feature_map]
    return groups


# Pruning strategy based on TimeSHAPs temporal coalition pruning algorithm
def find_temporal_prune_depth(
    *,
    model: ExternalModel,
    dataset: DataSet,
    location: str,
    horizon: int,
    num_perturbations: int,
    sampler_name: str,
    last_n: int | None,
    seed: int | None,
    timed: bool,
    start: float,
    threshold: float,
) -> tuple[int, list[float]]: #TODO change return value later after testing
    
    max_depth = len(dataset.period_range)
    print(f"Starting temporal pruning, max depth={max_depth}")

    all_shap_values = []

    # Ranges fdrom max_depth -1 to 1
    for split_index in range(max_depth - 1, 0, -1):
        print(f"    At split index {split_index}")

        """
        TODO regarding segmentation REMEBER TO REMOVE
        I need to pass down arguemtns to the segmenter from this call since I need to call it multiple times
        My solution now is to change segmenter argument to also accept a object of segmentation model
        Issues: Comlpicated code, and I need to rerun data preperation for each time I want to change the segmentation
        Solution: I will refactor it later to first do data prerperation -> call segment() that does the segmentation.
        And mby refactor segmentation part further
        """
        inputs = prepare_explain_inputs(
            dataset=dataset,
            location=location,
            horizon=horizon,
            segmenter=SplitSegmentation(split_index=split_index),
            granularity=1,
            sampler_name=_check_allowed_sampler(sampler_name),
            seed=seed,
            last_n=last_n,
            timed=timed,
            start=start,
        )

        feature_map = build_feature_map(inputs.x0)
        feature_map_no_static = drop_static_features(feature_map)

        groups = group_two_player_split(feature_map=feature_map_no_static)

        result = _explain_shap(
            model=model,
            dataset=dataset,
            location=location,
            horizon=horizon,
            num_perturbations=num_perturbations,
            seed=seed,
            inputs=inputs,
            feature_map=feature_map_no_static,
            groups=groups,
            player_names=["new", "old"],
        )

        values = dict(result)
        old_val = values["old"]

        all_shap_values.append(old_val)
        # new_val = values["new"]
        # total = (abs(old_val) + abs(new_val))
        # old_share = abs(old_val) / total if total > 0 else 0.0

        print(f"    SHAP value for split_index: {old_val}")

        """
        TODO threshold is compared to the real shap value, so how much the prediciton was moved.
        This means that the caller needs to know the model and its predicitons well enough
        to pick a suitable trheshhold.
        Solution: Calculate the value threshold is comapred to based on data/predicitons, feks give
        old window moved it by 5% percent over
        Problems: Model can be probabilistc, so a lot of noice between calls that are not from the split
        """
        if abs(old_val) < threshold:
            #return split_index
            continue # TODO just for testing

    return 0, all_shap_values
