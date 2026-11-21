# Addition C -> T -> Y: MCQA-style layer pipeline

This pipeline follows **SDAS full layer training -> CAL evaluation and promising
layer selection -> GDAS training ONLY on those candidates**. Chain-level
test evaluation comes after training, not before GDAS.

- **C** = carry from ones column
- **T** = tens digit after carry
- **Y** = final numeric answer (possibly multi-token)

The original `addition_train_standard_grid.py` and
`addition_train_gradual_grid.py` are left intact. The new files reuse
their loss functions, handle trainers, readouts, and chain evaluation.

## Quick start (run in this exact order)

```bash
# 1. Train SDAS, all blocks for both variables
python3 addition_train_all_layers_standard.py \
  --result-dir results_addition_standard_all_layers_seed0 \
  --layers all --variables T,C --seed 0

# 2. Evaluate CAL IIA for every SDAS handle, print/save promising C/T layers
python3 addition_eval_all_layers_standard.py \
  --result-dir results_addition_standard_all_layers_seed0 \
  --threshold-pct 5

# 3. Evaluate all 20 SDAS-selected configurations on held-out TEST
python3 addition_eval_selected.py \
  --method standard \
  --result-dir results_addition_standard_all_layers_seed0

# 4. Train GDAS ONLY on the same candidate configurations; freeze local T copies
python3 addition_train_gradual_selected.py \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_selected_seed0 \
  --lambda-med 2 --lambda-readout 2 --seed 0

# 5. Evaluate GDAS from its own self-contained checkpoints (no Standard root)
python3 addition_eval_selected.py \
  --method gradual \
  --result-dir results_addition_gradual_selected_seed0 \
  --seed 0
```

The step-2 evaluation uses the **validation/CAL split only**; it does not
touch test pairs. The final step-4 evaluations use held-out TEST.

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

## 2. Evaluate ALL Standard layers on CAL, then select promising candidates

```bash
python3 addition_eval_all_layers_standard.py \
  --result-dir results_addition_standard_all_layers_seed0 \
  --threshold-pct 5
```

This file **actually reloads and evaluates every saved SDAS handle**,
rather than merely ranking the training log. For each layer it runs
free-autoregressive interventions on the CAL validation bank, recomputes
numeric output IIA, checks C/T readout accuracy, and writes:

```text
results_addition_standard_all_layers_seed0/
  calibration_standard/
    C_layer_00.json
    T_layer_00.json
    ...
    summary.csv              # CAL IIA + clean readout for every layer
  selected_layers_standard.json
  selected_layers_standard.csv
```

Select each variable's layers using the MCQA rule:

```
CAL_IIA(layer) >= max_CAL_IIA(variable) - 0.05
```

Five percentage **points**, not 5% relative. Take the Cartesian
product of the selected C/T layers, retaining only configurations
where `C < T`. Both the layer lists and valid pairs are printed.

**Step 3 GDAS reads this saved selection by default.** It validates
that candidate checkpoints/readouts are still identical to those
evaluated in step 2. If you change them, rerun the CAL eval first.

## 3. Final Standard DAS configuration evaluation (optional, after GDAS training)

```bash
python3 addition_eval_selected.py \
  --method standard \
  --result-dir results_addition_standard_all_layers_seed0
```

This does **not** retrain. It evaluates every selected (C,T) configuration
on held-out TEST `T_only`, `C_only` and `C_T` banks. Metrics include:

- clean C/T readout
- T direct intervention: `T_after_T`, `Y_after_T`, joint accuracy
- C intervention: `C_after_C`, `T_after_C`, `Y_after_C`, joint accuracy
- Y through frozen T mediator replay: `Y_via_T_replay`
- T restoration: `Y_restored_via_T`

### Different-recipient transfer (also included by default)

The evaluator also tests **cross-question recipients** as in MCQA:
each saved donor intervention is transferred to a distinct, model-correct
TEST prompt with a different ones digit. The target `Y` is computed on
the **recipient** arithmetic context, never copied from the donor base.

For each donor family `T_only`, `C_only`, and `C_T`:
- `Y_via_T_replay_recipient_same/different`: replace T on a different
  question with the donor's source T (`T_only`) or its C-induced T
  (`C_only`, `C_T`). Here same/different refers to recipient's
  natural T versus the patched T. Correct Y = `10 * patched_T + recipient_ones`.
- For `C_only` and `C_T` also measure
  `T_after_C_recipient_same/different`,
  `Y_after_C_recipient_same/different`, and
  `Joint_T_Y_after_C_recipient_same/different`:
  patch C into a different question and evaluate the resulting T and
  free-generated Y. Here same/different refers to recipient's
  natural C versus donor C. The expected new tens digit is
  `recipient_T - recipient_C + donor_C`.

Recipients are assigned **deterministically** from model-correct
filtered TEST prompts (separate from the original donor/base prompt),
and are identical for SDAS and GDAS when using the same dataset/seed.
The assignments cannot be selected via TEST accuracy.

Previous `metrics.json` files from an evaluation run are **upgraded**
without recomputing direct/replay/restoration: rerun the SAME
`addition_eval_selected.py` command after updating the branch. Use
`--seed 0` for deterministic recipient sampling, and `--skip-recipient`
to opt out of this extra evaluation.

Recipient results are saved in `metrics.json["recipient"]` and as flat
columns such as `C_only_Y_via_T_replay_recipient_different` in
`evaluation_standard/summary.csv` or
`evaluation_gradual/summary.csv`.

Each configuration goes to
`evaluation_standard/C_00_T_23/metrics.json`; combined metrics go to
`evaluation_standard/summary.csv`. Evaluations resume by checking
exact artifact/data/generation signatures.

## 4. GDAS: train C conditioned on each candidate T

```bash
python3 addition_train_gradual_selected.py \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_selected_seed0 \
  --lambda-med 2 \
  --lambda-readout 2 \
  --seed 0
```

By default, train GDAS **only on (C,T) candidate pairs saved by the Step-2 CAL eval**.
The exact frozen **T checkpoint and T readout are copied byte-for-byte**
from Standard into the GDAS output directory **before C training**.
GDAS C training and evaluation then load these local immutable T snapshots
(no independent/retrained T in GDAS). C uses direct Y sequence loss,
a frozen-T mediator/replay loss, and a T readout loss.

The GDAS C handle is **conditional on T**: a C at layer 5 trained
against T layer 23 is distinct from a C at layer 5 trained against
T layer 24. Checkpoints are saved accordingly:

```text
results_addition_gradual_selected_seed0/
  run_manifest.json
  training_pairs.json                # original CAL-selected (C,T) grid
  summary.csv
  T_handles/T_layer_23/
    das_t.pt                         # identical copy of SDAS frozen T
    t_readout_LogisticRegression.pt
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

## 5. Evaluate GDAS on the same selected configurations, independently

```bash
python3 addition_eval_selected.py \
  --method gradual \
  --result-dir results_addition_gradual_selected_seed0 \
  --seed 0
```

The GDAS evaluation reads **only** its own local C checkpoints, local
byte-identical frozen T snapshots and `training_pairs.json`.
It does **not** access `results_addition_standard_all_layers_seed0`.
The `training_pairs.json` is sealed during GDAS training so that
Standard and Gradual results are matched on all 20 Standard CAL-selected
pairs (or however many were selected). The evaluator checks
`train_signature.json` against each frozen T snapshot's SHA-256 hash
to prevent evaluating C against the wrong T handle.

### Backfill an existing GDAS run (no GPU training)

Runs trained **before** the T snapshot change can be made self-contained
without retraining or regenerating metrics:

```bash
python3 addition_gradual_snapshot.py \
  --standard-root results_addition_standard_all_layers_seed0 \
  --result-dir results_addition_gradual_selected_seed0

python3 addition_eval_selected.py \
  --method gradual \
  --result-dir results_addition_gradual_selected_seed0 \
  --seed 0
```

The first command loads **no model**, copies only the small frozen T
checkpoint/readout artifacts and writes `training_pairs.json`. It verifies
that their hashes match the original T artifacts recorded when each GDAS C
was trained; it rejects mismatches rather than silently replacing T.
Existing cached results are reused by checkpoint **content hash**, even
though the T file path has changed.

An **optional**, separate GDAS-specific candidate analysis is available
via `addition_select_promising_layers.py --method gradual`, which first
retains T based on the frozen Standard T CAL IIA and then retains C
conditionally within each T layer based on GDAS direct numeric CAL IIA.
To evaluate that *different subset*, explicitly pass
`--selection results_addition_gradual_selected_seed0/selected_layers_gradual.json`.
No TEST scores enter either layer selection.

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
