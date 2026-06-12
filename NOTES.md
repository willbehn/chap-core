# 2026-06-12 — William: Review of Behdad's stacked ensemble PR

Context: [PR #325 "Stacking new"](https://github.com/dhis2-chap/chap-core/pull/325).
Plain-language guide to what the PR does: [STACKED_ENSEMBLE_README.md](../../chap-core/STACKED_ENSEMBLE_README.md).
(File links below point to the PR's head commit, so they stay valid even if the branch moves.)

**Weaknesses of Behdad's code:**

- Poor code quality overall — hard to read.
- [`EnsembleModel.train()`](https://github.com/dhis2-chap/chap-core/blob/ec3c75c86ee8e19309291efeaeb09014a7b49ca5/chap_core/ensemble/ensemble_model.py)
  is the weak point. It branches on `if method == "probabilistic" / else` all the
  way through (building features, cleaning NaNs, fitting the meta-model), so the
  ensemble strategy is baked into the training loop. There is no way to extend
  it beyond stacking without adding yet more branches.
- The same `train()` function simply does too much: inner dataset splitting,
  training base models, extracting predictions, NaN handling, weight fitting,
  and the final retrain — all in one ~100-line method. The dataset-splitting
  logic in particular is complex and should be extracted into its own
  function/class.
- The meta-models in
  [`_meta_models.py`](https://github.com/dhis2-chap/chap-core/blob/ec3c75c86ee8e19309291efeaeb09014a7b49ca5/chap_core/ensemble/_meta_models.py)
  (`NonNegativeMetaModel`, `ProbabilisticMetaModel`) already share the same
  shape — both have `fit()` and `predict()` — but don't share a common
  superclass/interface. Giving them one would make meta-models interchangeable,
  so new strategies can be added as new classes instead of new branches in the
  train loop. This matters concretely: Ole will implement Boosting, and that
  must not become another `elif` in `train()`.
- The `method` parameter (default `"probabilistic"`) gates most of the logic in
  `train()`, even though the information is already implied by which meta-model
  you pass in (`ProbabilisticMetaModel` vs `NonNegativeMetaModel`). The mode
  should be inferred from the meta-model object instead of being a separate
  string flag that can disagree with it.
- The new CLI command in
  [`cli_endpoints/ensemble.py`](https://github.com/dhis2-chap/chap-core/blob/ec3c75c86ee8e19309291efeaeb09014a7b49ca5/chap_core/cli_endpoints/ensemble.py)
  is a separate `evaluate-ensemble` command. It should instead be integrated
  into the official `chap eval` command, so ensembles are evaluated the same
  way as any other model. (Note: as the PR stands the command isn't even
  reachable — [`cli.py`](https://github.com/dhis2-chap/chap-core/blob/master/chap_core/cli.py)
  is never updated to register it.)

**Plan:**

- [ ] Decide: merge out of Behdad's fork at this point? (Take the working core,
      refactor the structure ourselves, rather than iterating in his branch.)
