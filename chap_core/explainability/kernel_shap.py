"""
Kernel Shap over data og lags representasjonen fra LIME pipelinen fra LIME. Work in progress
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import shap

from chap_core.explainability.lime import (
    build_feature_map,
    perturb_vectors,
    prepare_explain_inputs,
    print_time,
    produce_lime_dataset,
)
from chap_core.explainability.surrogate import SurrogateResult
from chap_core.models.external_model import ExternalModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chap_core.explainability.plot import plot_shap_values

logger = logging.getLogger(__name__)

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
    save: bool = False,
    plot: bool = True,
    plot_path: Path | None = None,
) -> list[tuple[str, float]]:

    start = time.perf_counter()
    if timed:
        logger.info("Started KernelSHAP pipeline")

    # Tar rådata + andre parametere og lager et ExplainInputs objekt med lag-features++
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

    feature_map = build_feature_map(inputs.x0)
    feature_names = [name for name, _, _ in feature_map]

    def predict_from_masks(mask_matrix: np.ndarray) -> np.ndarray:
        masks = [np.asarray(row) for row in np.atleast_2d(mask_matrix)]
        perturbations, perturbation_masks = perturb_vectors(
            inputs.hist_df,
            inputs.x0,
            inputs.feat_indices,
            inputs.sampler,
            feature_map,
            masks,
            global_means=inputs.global_means,
        )
        _, y, _, _ = produce_lime_dataset(
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
            full_dataset=dataset,
            full_future_weather=inputs.full_future_weather,
        )

        return np.asarray(np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None)))

    # Setter alle features av, flat array med 0
    background = np.zeros((1, len(feature_names)))

    # Masken for all input
    instance = np.ones((1, len(feature_names)))

    if seed is not None:
        np.random.seed(seed)

    # Lager en explainer med predict_from_masks som verdi funksjon
    explainer = shap.KernelExplainer(predict_from_masks, background)

    # Selve utregning av shap-verdier, går igjennom alle pertubrasjoner og kaller 
    # predict_from_mask på hver som regner ut modellens output for den pertubrasjonen.
    # Innad i shap_values() vil den fitte en linear regressjons modell som oppdateres for pertubarasjon, 
    # der den vekter de ulike featurene basert på hva som ga mest utslag
    shap_values = explainer.shap_values(
        instance,
        nsamples=num_perturbations,
        l1_reg=f"num_features({len(feature_names)})",
        silent=True,
    )

    if timed:
        print_time(start, "Finished KernelSHAP explanation in %.4f seconds")

    values = np.ravel(np.asarray(shap_values, dtype=float))
    base_value = float(np.ravel(np.asarray(explainer.expected_value))[0])

    sorted_results = SurrogateResult(feature_names=feature_names, weighting=values).as_sorted()

    # Idk ikke skjønt helt hvorfor vi er i log1p space
    logger.info(f"SHAP base value (log1p space): {base_value:+.4f}")
    logger.info("SHAP values:")
    for name, c in sorted_results:
        logger.info(f"{name:>12}: {c:+.4f}")

    if plot:
        base = Path(plot_path) if plot_path is not None else Path(f"shap_{location}.png")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = base.with_name(f"{base.stem}_{stamp}{base.suffix}")
        plot_shap_values(sorted_results, base_value, out_path)
        logger.info(f"Saved SHAP plot to {out_path}")

    if save:
        logger.error("Ikke implementert lagring av resultat")

    return sorted_results
