"""Export the missing A100 prefix plan; collect on four isolated GPUs; stage a merge.

Never writes the original cache or pilot. Collection uses the original native
Target codec and records actual producer hardware. A staged merge is not an
authorization to restart training or a claim of A100-native acquisition.
"""
import argparse
from hashlib import sha256
import gzip
import importlib
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import subprocess

from program62_data import atomic_json, checked_file, digest_file, jsonl, read_json, require, rollout_examples

HERE = Path(__file__).resolve().parent
FIRST = 478
TOTAL = 1247237
SHARD = 256


def immutable(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f"existing export differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".pending")
        pending.write_bytes(data)
        pending.replace(path)


def seal(path, value):
    immutable(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def teacher_identity(cfg):
    keys = ("backend_sha256", "model_manifest_sha256", "train_index_sha256", "expected_teacher_rows",
            "teacher_shard_rows", "device")
    value = {key: cfg[key] for key in keys}
    for key in ("program62_data.py", "rollout_coverage.py"):
        value[key] = cfg["implementation_sha256"][key]
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_check(cfg):
    require((cfg["expected_teacher_rows"], cfg["teacher_shard_rows"], cfg["device"]) == (TOTAL, SHARD, "cuda:0"),
            "not the original candidate-prefix supervision plan")
    for key, value in read_json(HERE / "candidate.json").items():
        require(cfg[key] == value, f"original source differs: {key}")
    for name, expected in cfg["implementation_sha256"].items():
        checked_file(HERE / name, expected)


def export_plan(config_path, out):
    cfg = read_json(config_path)
    source_check(cfg)
    out = out.resolve()
    cache = Path(cfg["teacher_root"]).resolve()
    require(not out.is_relative_to(cache) and not cache.is_relative_to(out)
            and not out.is_relative_to(HERE), "export overlaps original cache/source")
    checked_file(cfg["train_index"], cfg["train_index_sha256"])
    checked_file(cfg["model_manifest"], cfg["model_manifest_sha256"])
    require({p.name for p in cache.glob("part-*.json")} == {f"part-{p:08d}.json" for p in range(FIRST)},
            "original cache is no longer the frozen 478-part source")
    old, planned = [], []
    examples = iter(rollout_examples(cfg["train_index"], cfg["raw_shards"]))
    total = 0
    source_sha = teacher_identity(cfg)
    for number in range((TOTAL + SHARD - 1) // SHARD):
        rows = list(islice(examples, SHARD))
        require(len(rows) == min(SHARD, TOTAL - total), "frozen source plan row count differs")
        total += len(rows)
        if number < FIRST:
            path = cache / f"part-{number:08d}.json"
            receipt = read_json(path)
            require(receipt["part"] == number and receipt["rows"] == SHARD
                    and receipt["config_sha256"] == source_sha
                    and receipt["plan_sha256"] == cfg["train_index_sha256"], "original shard binding differs")
            checked_file(cache / receipt["rows_path"], receipt["rows_sha256"])
            saved = list(jsonl(cache / receipt["rows_path"]))
            require([r["prefix_sha256"] for r in saved] == [r["prefix_sha256"] for r in rows],
                    "478-part source is not this exact ordered plan")
            old.append(dict(receipt=receipt, receipt_sha256=digest_file(path)))
            if number == 0:
                for name, expected in ((receipt["rows_path"], receipt["rows_sha256"]),
                                       (receipt["hidden_path"], receipt["hidden_sha256"])):
                    checked_file(cache / name, expected)
                    immutable(out / "probe" / name, (cache / name).read_bytes())
        else:
            blob = gzip.compress(("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)).encode(), compresslevel=1, mtime=0)
            name = f"parts/part-{number:08d}.jsonl.gz"
            immutable(out / name, blob)
            planned.append(dict(part=number, worker=(number - FIRST) % 4, rows=len(rows),
                                path=name, sha256=sha256(blob).hexdigest()))
        if number % 128 == 0:
            print(json.dumps(dict(phase="export", part=number, source_rows=total)), flush=True)
    require(next(examples, None) is None and total == TOTAL, "source plan has extra rows")
    immutable(out / "models.json", Path(cfg["model_manifest"]).read_bytes())
    seal(out / "source_config.json", cfg)
    plan = dict(status="export_complete_not_collection_complete", total_rows=TOTAL,
                existing_rows=FIRST * SHARD, missing_rows=TOTAL - FIRST * SHARD,
                source_config_sha256=digest_file(out / "source_config.json"),
                source_index_sha256=cfg["train_index_sha256"], teacher_config_sha256=source_sha,
                model_manifest_sha256=cfg["model_manifest_sha256"], export_tool_sha256=digest_file(__file__),
                existing=old, planned=planned, probe_rows=8,
                assignment="(global_part-478) modulo4; independent output directories; never share a writer")
    seal(out / "plan.json", plan)
    print(json.dumps(dict(phase="export_complete", parts=len(planned), missing_rows=plan["missing_rows"])), flush=True)


def read_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    require(plan["status"] == "export_complete_not_collection_complete" and plan["total_rows"] == TOTAL
            and plan["missing_rows"] == TOTAL - FIRST * SHARD, "wrong export plan")
    require([p["part"] for p in plan["planned"]] == list(range(FIRST, (TOTAL + SHARD - 1) // SHARD))
            and all(p["worker"] == (p["part"] - FIRST) % 4 for p in plan["planned"]), "invalid four-worker partition")
    checked_file(path.parent / "source_config.json", plan["source_config_sha256"])
    cfg = read_json(path.parent / "source_config.json")
    source_check(cfg)
    require(teacher_identity(cfg) == plan["teacher_config_sha256"], "export teacher identity differs")
    return path, plan, cfg


def plan_rows(base, item):
    require(item["path"] == f"parts/part-{item['part']:08d}.jsonl.gz", "invalid plan shard path")
    path = base / item["path"]
    checked_file(path, item["sha256"])
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines() if line.strip()]
    require(len(rows) == item["rows"], "plan shard row count differs")
    return rows


def receipt_files(root, receipt):
    stem = f"part-{receipt['part']:08d}"
    require(receipt["rows_path"] == stem + ".jsonl" and receipt["hidden_path"] == stem + ".safetensors",
            "invalid output shard path")
    checked_file(root / receipt["rows_path"], receipt["rows_sha256"])
    checked_file(root / receipt["hidden_path"], receipt["hidden_sha256"])


def probe_original(plan_path, plan, target, data):
    from safetensors import safe_open
    receipt = plan["existing"][0]["receipt"]
    root = plan_path.parent / "probe"
    receipt_files(root, receipt)
    rows = list(jsonl(root / receipt["rows_path"]))[:plan["probe_rows"]]
    with safe_open(root / receipt["hidden_path"], framework="pt", device="cpu") as tensors:
        for row in rows:
            patch = dict(row["teacher_logit_patch"])
            key = row["hidden_key"]
            if patch["kind"] != "hidden":
                patch["values"] = tensors.get_tensor(key + ".logit_values")
            if patch["kind"] == "sparse":
                patch["ids"] = tensors.get_tensor(key + ".logit_ids")
            data.rebuild_teacher_logits(dict(row, _teacher_logit_patch=patch), tensors.get_tensor(key), target)
    return len(rows)


def collect(plan_path, worker, gpu_uuid, models_root, gpu_lock, out):
    path, plan, cfg = read_plan(plan_path)
    require(0 <= worker < 4 and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid and gpu_uuid.startswith("GPU-"),
            "each worker must expose exactly its assigned physical UUID")
    actual = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    require(actual == gpu_uuid, "GPU UUID differs")
    out = out.resolve()
    require(not out.is_relative_to(HERE) and not out.is_relative_to(path.parent), "worker output overlaps code/plan")
    identity = dict(plan_sha256=digest_file(path), worker=worker, gpu_uuid=gpu_uuid,
                    tool_sha256=digest_file(__file__), teacher_config_sha256=plan["teacher_config_sha256"])
    seal(out / "worker.json", identity)
    cfg = dict(cfg, models_root=str(models_root.resolve()), model_manifest=str(path.parent / "models.json"),
               source_root=str(HERE / "vendor"), decoder=str(HERE / "vendor/closed_loop_core.py"), gpu_lock=str(gpu_lock.resolve()))
    runtime = importlib.import_module("program62_run")
    data = importlib.import_module("program62_data")
    import transformers
    from safetensors.torch import save_file
    require(transformers.__version__ == "4.57.3", "use a separate pinned4.57.3 environment")
    require(runtime.torch.cuda.device_count() == 1, "worker sees multiple GPUs")
    assigned = [p for p in plan["planned"] if p["worker"] == worker]
    with runtime.worker_lease(cfg, out):
        target, _ = runtime.load_models(cfg, collect_only=True)
        with runtime.torch.inference_mode():
            probe_count = probe_original(path, plan, target, data)
        seal(out / "compatibility.json", dict(**identity, status="pass", rows=probe_count,
             scope="A100 sealed8 head/codec rows decoded on4090; reverse direction not yet measured"))
        count = 0
        for item in assigned:
            stem = f"part-{item['part']:08d}"
            terminal = out / (stem + ".json")
            rows = plan_rows(path.parent, item)
            if terminal.exists():
                receipt = read_json(terminal)
                require(receipt["producer"] == identity and receipt["rows"] == len(rows)
                        and receipt["part"] == item["part"] and receipt["prefix_plan_sha256"] == item["sha256"]
                        and receipt["config_sha256"] == plan["teacher_config_sha256"]
                        and receipt["plan_sha256"] == cfg["train_index_sha256"], "worker resume source differs")
                receipt_files(out, receipt)
            else:
                require(shutil.disk_usage(out).free > 2 * 1024**3, "less than2GiB disk reserve")
                values = {}
                for offset, row in enumerate(rows):
                    require(data.prefix_sha(row["prefix_ids"]) == row["prefix_sha256"], "prefix token binding differs")
                    with runtime.torch.inference_mode():
                        output, states = target._forward_with_final_hidden(target._ids(row["prefix_ids"]))
                        hidden = states[0, -1].contiguous().detach()
                        native = output.logits[0, -1]
                        rebuilt = target.model.get_output_embeddings()(hidden.reshape(1, -1))[0]
                        patch = data.exact_logit_payload(native, rebuilt)
                        require(patch["kind"] != "dense", "compact codec requires dense fallback; stop rather than produce342GB")
                        data.rebuild_teacher_logits(dict(row, _teacher_logit_patch=patch), hidden, target)
                        row["teacher_logit_patch"] = {k: v for k, v in patch.items() if k not in ("ids", "values")}
                        row["hidden_key"] = str(offset)
                        row["target_greedy_id"] = int(native.argmax())
                        values[str(offset)] = hidden.cpu()
                        for key in ("ids", "values"):
                            if key in patch:
                                values[f"{offset}.logit_{key}"] = patch[key]
                tensors = out / (stem + ".safetensors")
                pending = tensors.with_suffix(".safetensors.pending")
                save_file(values, str(pending))
                immutable(out / (stem + ".jsonl"), ("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)).encode())
                pending.replace(tensors)
                receipt = dict(part=item["part"], rows=len(rows), plan_sha256=cfg["train_index_sha256"],
                    config_sha256=plan["teacher_config_sha256"], rows_path=stem + ".jsonl",
                    rows_sha256=digest_file(out / (stem + ".jsonl")), hidden_path=tensors.name,
                    hidden_sha256=digest_file(tensors), producer=identity, prefix_plan_sha256=item["sha256"])
                seal(terminal, receipt)
            count += item["rows"]
            atomic_json(out / "status.json", dict(**identity, phase="collect", sealed_rows=count, last_global_part=item["part"]))
        seal(out / "completion.json", dict(**identity, status="complete_assigned_parts_only", rows=count,
             parts=len(assigned), full_original_collection_complete=False))


def link_file(source, dest, expected):
    checked_file(source, expected)
    if dest.exists():
        checked_file(dest, expected)
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)
    checked_file(dest, expected)


def merge(plan_path, source_cache, workers, out):
    path, plan, cfg = read_plan(plan_path)
    source_cache, out = source_cache.resolve(), out.resolve()
    require(not out.is_relative_to(source_cache) and not source_cache.is_relative_to(out)
            and not out.is_relative_to(HERE) and not out.is_relative_to(path.parent), "merge overlaps original data/source")
    require(len(workers) == 4 and len({p.resolve() for p in workers}) == 4, "need four independent worker directories")
    identities = []
    for i, root in enumerate(workers):
        identity = read_json(root / "worker.json")
        terminal = read_json(root / "completion.json")
        require(identity["worker"] == i and identity["plan_sha256"] == digest_file(path)
                and terminal["status"] == "complete_assigned_parts_only"
                and all(terminal[k] == v for k, v in identity.items()), "worker terminal differs")
        identities.append(identity)
    require(len({r["gpu_uuid"] for r in identities}) == 4, "workers share a physical GPU")
    out.mkdir(parents=True, exist_ok=True)
    for old in plan["existing"]:
        receipt = old["receipt"]
        source = source_cache / f"part-{receipt['part']:08d}.json"
        checked_file(source, old["receipt_sha256"])
        for key in ("rows", "hidden"):
            link_file(source_cache / receipt[key + "_path"], out / receipt[key + "_path"], receipt[key + "_sha256"])
        seal(out / source.name, receipt)
    for item in plan["planned"]:
        root = workers[item["worker"]]
        receipt = read_json(root / f"part-{item['part']:08d}.json")
        require(receipt["part"] == item["part"] and receipt["rows"] == item["rows"]
                and receipt["producer"] == identities[item["worker"]]
                and receipt["config_sha256"] == plan["teacher_config_sha256"]
                and receipt["plan_sha256"] == cfg["train_index_sha256"]
                and receipt["prefix_plan_sha256"] == item["sha256"], "collected shard source differs")
        receipt_files(root, receipt)
        expected = plan_rows(path.parent, item)
        actual = list(jsonl(root / receipt["rows_path"]))
        require(len(expected) == len(actual) and all(all(a[k] == v for k, v in e.items()) for a, e in zip(actual, expected)),
                "collected row differs from frozen A100 prefix plan")
        for key in ("rows", "hidden"):
            link_file(root / receipt[key + "_path"], out / receipt[key + "_path"], receipt[key + "_sha256"])
        seal(out / f"part-{item['part']:08d}.json", receipt)
    seal(out / "completion.json", dict(status="complete", rows=TOTAL, parts=(TOTAL + SHARD - 1) // SHARD,
         config_sha256=plan["teacher_config_sha256"], plan_sha256=cfg["train_index_sha256"],
         offload_plan_sha256=digest_file(path), producers=identities,
         scope="478 original A100 parts plus4090 native labels on identical frozen prefixes; collection coverage only",
         A100_reverse_codec_readback_measured=False, training_started_by_this_tool=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--config", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    col = sub.add_parser("collect")
    col.add_argument("--plan", type=Path, required=True)
    col.add_argument("--worker", type=int, required=True)
    col.add_argument("--gpu-uuid", required=True)
    col.add_argument("--models-root", type=Path, required=True)
    col.add_argument("--gpu-lock", type=Path, required=True)
    col.add_argument("--out", type=Path, required=True)
    mer = sub.add_parser("merge")
    mer.add_argument("--plan", type=Path, required=True)
    mer.add_argument("--source-cache", type=Path, required=True)
    mer.add_argument("--workers", type=Path, nargs=4, required=True)
    mer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "export":
        export_plan(args.config, args.out)
    elif args.phase == "collect":
        collect(args.plan, args.worker, args.gpu_uuid, args.models_root, args.gpu_lock, args.out)
    else:
        merge(args.plan, args.source_cache, args.workers, args.out)


if __name__ == "__main__":
    main()
