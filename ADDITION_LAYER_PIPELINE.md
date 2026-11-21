# Addition C -> T -> Y: MCQA-style layer pipeline

This is a **train -> select -> evaluate** pipeline for the two-variable
causal abstraction `C -> T -> Y`, patterned after the MCQA
`mcqa_train_all_layers_no_eval.py` / `mcqa_eval_saved_all_layers.py`
and train-only GDAS workflow.

- **C** = carry from ones column
- **T** = tens digit after carry
- **Y** = final numeric answer (possibly multi-token)

The original `addition_train_standard_grid.py` and
`addition_train_gradual_grid.py` are left intact. The new files reuse
their loss functions, handle trainers, frozen readouts, and causal-chain
evaluation functions.

## 0. Prepare data

Generate `addition_controlled_data/{train,validation,test}.jsonl` using
`python3 addition_generation.py` if they do not exist yet.
`addition_constants.py` (model name and HF token) stays local, as in
the existing repo. The `addition_data_load.py` filtering cache is
automatically reused when its data/model manifest matches.

## 1. Standard DAS: train ALL layers, no test evaluation

```bash
python3 addition_train_all_layers_standard.py \
  --result-dir results_addition_standard_all_layers_seed0 \
  --variables T,C \
  --layers all \
  --seed 0
```

By default the sweep uses **all model block indices** (`0..N-1`),
trains independent T handles using `T_only` and independent C handles
using `C_only`, and trains their frozen linear readouts. It saves:

```text
results_addition_standard_all_layers_seed0/
  run_manifest.json
  summary.csv                  # variable,layer,cal_iia,readout_cal_acc,...
  T_handles/T_layer_00/das_t.pt
  T_handles/T_layer_00/t_readout_LogisticRegression.pt
  ...
  C_handles/C_layer_00/das_c.pt
  C_handles/C_layer_00/c_readout_LogisticRegression.pt
```

`cal_iia` is the validation **free-autoregressive numeric output IIA**
(saved at each handle's best epoch), not teacher-forced token accuracy.
No test pairs are used for training or layer selection.

The command is resumable: existing valid checkpoints/readouts are
skipped. Subsets are supported, e.g. `--variables T --layers 12:26`
or `--variables C --layers 0,1,2`. Reuse the same result directory
only with identical seed, model, data and training hyperparameters.
Changing those requires a new result directory.

## 2. Choose promising Standard layers

```bash
python3 addition_select_promising_layers.py \
  --method standard \
  --result-dir results_addition_standard_all_layers_seed0 \
  --threshold-pct 5
```

Per variable, keep all layers satisfying

```
cal_iia(layer) >= max_cal_iia(variable) - 0.05
```

The threshold means **5 percentage points**, not 5% relative.
Take the Cartesian product and exclude invalid `C >= T` configurations.

Outputs:
- `selected_layers_standard.json`: C and T candidate layer lists plus valid pair list
- `selected_layers_standard.csv`: one row per selected (C,T) pair

## 3. Evaluate selected Standard configurations

```bash
python3 addition_eval_selected.py \
  --method standard \
  --result-dir results_addition_standard_all_layers_seed0
```

This **does not retrain**. It evaluates every saved (C,T) configuration
on test `T_only`, `C_only` and `C_T` banks. It reports:

- clean C/T readout accuracies
- T direct intervention: `T_after_T`, `Y_after_T`, joint accuracy
- C intervention: `C_after_C`, `T_after_C`, `Y_after_C`, joint accuracy
- C-mediated Y via frozen T replay: `Y_via_T_replay`
- restoration at frozen T: `Y_restored_via_T`

Each configuration is saved immediately to
`evaluation_standard/C_00_T_23/metrics.json`, and metrics are collected
in `evaluation_standard/summary.csv`. Run again to skip existing
evaluations, or pass `--force` to recompute. Saved results are reused
only if checkpoints, filtered data and generation settings still match.

## 4. GDAS: train C conditioned on each candidate T

```bash
python3 addition_train_gradual_selected.py \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_selected_seed0 \
  --lambda-med 2 \
  --lambda-readout 2 \
  --seed 0
```

By default, train GDAS on the **Standard-selected (C,T) pairs**.
The exact frozen **T checkpoint and T readout are reused from Standard**
(no independent/retrained T in GDAS). C uses direct Y sequence loss,
a frozen-T mediator/replay loss, and a T readout loss.

The GDAS C handle is **conditional on T**: a C at layer 5 trained
against T layer 23 is distinct from a C at layer 5 trained against
T layer 24. Checkpoints are saved accordingly:

```text
results_addition_gradual_selected_seed0/
  run_manifest.json
  summary.csv
  pairs/C_05_T_23/
    train_signature.json
    das_c.pt
    c_readout_LogisticRegression.pt
  pairs/C_05_T_24/
    train_signature.json
    das_c.pt
    c_readout_LogisticRegression.pt
```

To **exhaustively sweep GDAS** (all valid pairs, `C<T`), first train
all Standard T layers (step 1), then run:

```bash
python3 addition_train_gradual_selected.py \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_all_pairs_seed0 \
  --t-layers all --c-layers all --seed 0
```

This can be expensive: for an N-block model it trains
`N(N-1)/2` distinct conditional GDAS C handles, not just `2N`
independent handles. Run subsets with e.g. `--t-layers 23,24`
and `--c-layers 0:16` to reduce compute.

## 5. Choose promising GDAS pairs, then evaluate

```bash
python3 addition_select_promising_layers.py \
  --method gradual \
  --result-dir results_addition_gradual_selected_seed0 \
  --threshold-pct 5

python3 addition_eval_selected.py \
  --method gradual \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_selected_seed0
```

For GDAS, retain the best calibrated T layers (using the same frozen
Standard T calibration IIA); **within each T layer**, retain C layers
whose **GDAS direct numeric C IIA** is within 5 percentage points
of the best C for that T. This respects the conditional structure
of GDAS. The saved `C_cal_score` (minimum of direct, mediator and
readout validation scores) is a diagnostic, not the layer-selection
metric. No test scores enter selection.

Results are under
`evaluation_gradual/C_05_T_23/metrics.json` and
`evaluation_gradual/summary.csv` with the **same held-out metrics**
as the Standard evaluation.

## Validation and caveats

- Syntax and calibration-selection tests run in CI without a GPU.
- Full training/inference has not been GPU-run by this change; it
  needs the original repository's model, HF credential and data.
- The new pipeline intentionally rejects legacy/incompatible
  result directories instead of silently reusing old checkpoints.
- The underlying original evaluation code uses decoded-number IIA:
  `Y` may use several generated tokens. This is intentionally retained.
- Use `python3 addition_eval_selected.py --method standard
  --result-dir ... --list-only` to inspect the selected configurations
  before loading a model.
