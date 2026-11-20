# Four4090 continuation of the A100 candidate-prefix cache

The original478 A100 parts and the active pilot are preserved. This adds only
missing candidate-prefix teacher records. It does not use the separate4090
Task14 RAW trajectories, restart training, or publish a completion in the
original cache.

The worker reuses the original native Target forward and H+sparse-logit codec.
Dense fallback stops the worker; it does not silently create342GB of logits.
Actual producer UUIDs and the exported prefix plan bind every new part.
Transformers4.57.3 and the original model-file hashes remain required.

## 1. CPU export on A100

Run outside the original source/cache and the active pilot. This phase does
not load a model or use a GPU. It writes gzipped per-part prefix plans, the
source configuration/model manifest, and a small original-shard codec probe.

```sh
python /PATH/TO/CHECKOUT/offload_teacher.py export \
  --config /data/autoresearch/task70-single-kv-slot-results-20261120-74c69312/equal/config.json \
  --out /data/autoresearch/task73-prefix-plan-20261120
```

Transfer the entire exported directory to4090, plus this code checkout.
The plan has1,124,869 missing positions in4,395 global parts478..4872, with
the last part containing5 rows. Worker ownership is `(part-478) modulo4`.
Record the actual `plan.json` SHA after export; no export-complete receipt
means collection is complete.

## 2. Four independent collectors

Use the existing shared physical UUID lock for each card, as resolved by the
4090 machine operator. Do not invent a separate lock namespace. Each process
must expose exactly one UUID, use a distinct worker directory, and keep its
own log. Model-root and lock paths below are operator-resolved actual paths.

```sh
CUDA_VISIBLE_DEVICES=GPU-UUID /PINNED/4573/bin/python /CHECKOUT/offload_teacher.py collect \
  --plan /EXPORTED/PLAN/plan.json --worker 0 --gpu-uuid GPU-UUID \
  --models-root /BOUND/MODEL/PARENT --gpu-lock /EXISTING/SHARED/UUID.lock \
  --out /NEW/TASK73/worker-0
```

Repeat for worker1/2/3 on their own card/output/lock. The known4090 UUIDs from
the machine's last inventory are:

| Worker | UUID |
| --- | --- |
| 0 | GPU-cda31e51-fed9-d01d-8c62-78fe268974ac |
| 1 | GPU-0948a32e-fdb2-0020-1aa0-4a3b7b9920ba |
| 2 | GPU-beda3ec8-ee8e-7e00-e357-bee4438e9a11 |
| 3 | GPU-5f2d78f8-bcf5-5a25-cf0f-6683c1b1b23a |

The original-shard probe decodes8 sealed A100 H/codec rows on the assigned4090
and then collection continues directly. It is one format compatibility check,
not another full evaluation gate. A failure is reported rather than bypassed.
It does not prove the reverse direction for every new4090 row; the unchanged
A100 training reader still validates each native-distribution digest when
those data are later consumed. No second training run is started here.

## 3. Stage the completed collection on A100

After all four worker terminals, transfer their outputs back. Supply worker
directories in0,1,2,3 order and choose a new merge directory. The original
478 cache and pilot remain untouched. Local immutable files may use hard
links to avoid a second disk copy; no chmod or source writes are performed.

```sh
python /CHECKOUT/offload_teacher.py merge --plan /PLAN/plan.json \
  --source-cache /data/autoresearch/task62-a100-oracle-results-20261119/shared-teacher-exact \
  --workers /IMPORTED/worker-0 /IMPORTED/worker-1 /IMPORTED/worker-2 /IMPORTED/worker-3 \
  --out /data/autoresearch/task73-mixed-source-teacher-20261120
```

The merge checks coverage, file hashes, all exported row fields, disjoint
worker hardware identities, and records mixed acquisition hardware. Its
completion means collection coverage, not A100-native acquisition, a new
training admission, or proof of effect. Current pilot train/eval proceeds
independently.

Local delivery verification: the CLI parsed and displayed help. The new
orchestration has not run on actual GPUs; first actual part/compatibility
receipts must be reported as such. Existing codec math/source is unchanged.
