# Completed pilot: original503 comparison

User `49a8f371` requires the same full503 development roster as the earlier
heads and LoRA. Run TRUE first. Historical RAW acceptance counts are already
in the pinned roster; they are not fresh timing measurements.

After the existing pilot trainer exits successfully and writes its training
completion, use the same pinned environment and physical GPU1 UUID/lock:

```sh
CUDA_VISIBLE_DEVICES=GPU-2a4ca0b6-800e-7883-d5db-d250ed99523d \
  /data/autoresearch/task70-venv-4573/bin/python /PATH/TO/THIS/CHECKOUT/eval503.py \
  --training-root /data/autoresearch/task70-single-kv-slot-pilot478-results-20261120-24ac59c
```

Results go to `TRAINING_ROOT/eval503/true`. The adapter checks the completed
same-config checkpoint, keeps the training config and weights unchanged, and
calls the original native Target/Draft decoder. Warmup is excluded. Each
request seals independently and is validated again on resume.

`--mode mask` and `--mode shuffle` select the original diagnostic controls;
they write separate directories. Do not delay the TRUE503 main result until
all diagnostics finish. No new RAW run or pre-training parity is requested.

The roster SHA256 is
`a75803ceae25fcf21a4ae42cb87470bca375096d9e0676ed52645a727ae4a836`.
It is the identical 503-request file from the existing-data LoRA experiment,
including original order, prompts, Target tapes and historical count baselines.

Reports disclose the partial478 one-pass training budget, the different
historical head/LoRA budgets, and reuse of a development set. Acceptance length
uses accepted Draft tokens divided by verification rounds, excluding correction
and bonus tokens. Historical wall times do not establish a new speedup.

The entry was exercised once with all503 rows and the frozen decoder using
synthetic CPU backends; that is an executable-entry check, not a GPU/model
evaluation result. No additional model gate is required before the authorized
completed-checkpoint evaluation.
