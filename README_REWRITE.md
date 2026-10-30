# Addition causal abstraction rewrite

Causal model:

`C -> T -> Y`

- `C`: carry from the ones column.
- `T`: tens digit after adding the carry.
- `Y`: final addition answer.

## Important design choices

1. `Y` may tokenize into more than one token.
   - Training uses teacher-forced sequence cross-entropy because the loss must remain differentiable.
   - Validation/test IIA is autoregressive: after the shared answer-prefix token, the model generates the answer tokens itself under the intervention.
   - The primary `*_iia` metric compares the decoded generated number with the counterfactual integer target.
   - Token-exact and teacher-forced metrics are also saved as diagnostics.
   - Sequence CE is averaged within each example first, so examples with two answer tokens do not receive twice the weight of one-token examples.

2. Train/validation/test are prompt-disjoint.
   - The full addition problem pool is split first.
   - Base and all donor/source prompts for a row are sampled only from that split's pool.
   - The generator checks that no base/source prompt appears in two splits.
   - `--repeats-per-base 3` is used by default so there are enough controlled pairs after enforcing this stronger split.

3. Layer selection uses learned DAS, not full-vector intervention.
   - Sweep T layers by training a T DAS handle at each candidate layer.
   - Select T using validation autoregressive numeric IIA.
   - Fit the frozen T readout at the selected T layer.
   - Sweep C layers before T by training a C DAS handle at each candidate layer.
   - Select C using validation score.
   - Test is evaluated only after the T/C layer has been selected.

## File structure

- `addition_generation.py`: generate raw controlled data only.
- `addition_data_load.py`: load raw data, filter model-correct pairs, cache filtered pairs, tokenize, and build banks.
- `addition_das.py`: common DAS, sequence-loss, and autoregressive-intervention utilities.
- `addition_train_t.py`: train one fixed T layer and fit a T readout.
- `addition_train_c.py`: train one fixed C layer using a frozen T handle/readout.
- `addition_joint_das_c_t.py`: learned-DAS T sweep followed by learned-DAS C sweep.
- `addition_eval_readout_mediator.py`: final direct/readout/mediator/recipient evaluation.
- `addition_neural_network.py`: model/tokenizer loader only.
- `addition_sanity_check.py`: raw-data and filtered-pair invariant checks.

`addition_constants.py` remains local and ignored by git.

## 1. Generate data

```bash
python3 addition_generation.py
python3 addition_sanity_check.py
```

The generator writes:

```text
addition_controlled_data/
  train.jsonl
  validation.jsonl
  test.jsonl
```

## 2. Standard DAS sweep

```bash
python3 addition_joint_das_c_t.py \
  --result-dir results_addition_standard_sweep \
  --t-layers 16,17,18,19,20,21,22 \
  --c-layers 10,11,12,13,14,15,16,17,18 \
  --c-lambda-med 0.0 \
  --c-lambda-readout 0.0 \
  --generation-steps 4 \
  --seed 0
```

Use `--resume` only with checkpoints produced by this rewrite. Old checkpoints are rejected because their validation IIA used the old teacher-forced metric.

## 3. Gradual DAS sweep

```bash
python3 addition_joint_das_c_t.py \
  --result-dir results_addition_gradual_sweep \
  --t-layers 16,17,18,19,20,21,22 \
  --c-layers 10,11,12,13,14,15,16,17,18 \
  --c-lambda-med 2.0 \
  --c-lambda-readout 2.0 \
  --generation-steps 4 \
  --seed 0
```

## 4. Final evaluation

```bash
python3 addition_eval_readout_mediator.py --result-dir results_addition_standard_sweep --generation-steps 4 --seed 0
python3 addition_eval_readout_mediator.py --result-dir results_addition_gradual_sweep --generation-steps 4 --seed 0
```

The primary metrics `Y_after_T`, `Y_after_C`, and `Y_via_T_replay` are autoregressive decoded-number accuracies. Metrics ending in `_teacher_forced` are diagnostics only.
