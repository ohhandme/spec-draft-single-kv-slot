"""Frozen RAW rollout examples and resumable exact-prefix Target supervision."""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from rollout_coverage import draft_rows, prefix_sha


def digest_file(path):
    with Path(path).open("rb") as handle:
        from hashlib import file_digest
        return file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def checked_file(path, expected):
    require(digest_file(path) == expected, f"input digest differs: {path}")


def rollout_examples(index_path, shards_path):
    """Use every scored candidate position from the frozen training trajectories."""
    seen = set()
    for item in jsonl(index_path):
        if item["split"] != "train":
            continue
        rid = item["request_id"]
        require(rid not in seen and "/" not in rid and "\\" not in rid, "invalid RAW request ID")
        seen.add(rid)
        shard = Path(shards_path) / rid
        trace_path = shard / "trace.json"
        receipt = read_json(shard / "receipt.json")
        require(receipt["request_id"] == rid and receipt["repair_enabled"] is False,
                f"RAW identity differs: {rid}")
        require(receipt["all_four_groups_readback_pass"] is True, f"RAW shard not sealed: {rid}")
        checked_file(trace_path, receipt["files"]["trace.json"])
        trace = read_json(trace_path)
        require(trace["request_id"] == rid and trace["output_ids"] == item["output_ids"]
                and trace["prompt_ids"] == item["prompt_ids"], f"frozen request differs: {rid}")
        rows = iter(draft_rows(trace))
        committed = trace["prompt_ids"].copy()
        previous_feedback = None
        for round_info in trace["rounds"]:
            visible_from = len(committed) - 1
            proposal = round_info["candidate_ids"]
            for position in range(len(proposal)):
                row = next(rows)
                prefix = committed + proposal[:position]
                require(prefix_sha(prefix) == row["prefix_sha256"], "candidate predecessor differs")
                yield {
                    **row,
                    "family_id": trace["family_id"],
                    "prefix_ids": prefix,
                    "draft_depth": position + 1,
                    "visible_from": visible_from,
                    "previous_feedback": previous_feedback,
                }
            committed.extend(round_info["committed_ids"])
            if round_info["stop_reason"] != "continue":
                continue
            hidden = round_info["hidden"]
            require(hidden["present"] and hidden["available_before_receipt"]
                    and hidden["same_effective_target_forward_as_decision"]
                    and hidden["source_prefix_sha256"] == prefix_sha(committed[:-1])
                    and hidden["predicted_token_id"] == committed[-1],
                    f"feedback is off the legal verifier boundary: {rid}")
            previous_feedback = {
                "path": str(shard / "features.safetensors"),
                "request_id": rid,
                "file_sha256": receipt["files"]["features.safetensors"],
                "key": round_info["tensor_prefix"] + ".hidden",
                "hidden_sha256": hidden["sha256"],
                "accepted": round_info["accepted_count"],
                "generated": len(committed) - len(trace["prompt_ids"]),
            }
        require(next(rows, None) is None, f"coverage walk not exhausted: {rid}")


@lru_cache(maxsize=32)
def _feedback_file(path, expected):
    from safetensors.torch import load_file
    checked_file(path, expected)
    return {name: array for name, array in load_file(path, device="cpu").items()
            if name.endswith(".hidden")}


def feedback_hidden(source):
    import torch
    tensor = _feedback_file(source["path"], source["file_sha256"])[source["key"]]
    actual_sha = sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    require(actual_sha == source["hidden_sha256"] and bool(tensor.isfinite().all()),
            "previous-round hidden tensor binding differs")
    return tensor


def completed_teacher_shards(root, plan_sha, config_sha):
    root = Path(root)
    receipts = sorted(root.glob("part-*.json"))
    for number, path in enumerate(receipts):
        row = read_json(path)
        require(row["part"] == number, "supervision shards are not contiguous")
        require(row["plan_sha256"] == plan_sha and row["config_sha256"] == config_sha,
                "teacher shard source changed")
        checked_file(root / row["rows_path"], row["rows_sha256"])
        checked_file(root / row["hidden_path"], row["hidden_sha256"])
        yield row


def collect_teacher(config, config_sha, target, root):
    import itertools
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    index = config["train_index"]
    checked_file(index, config["train_index_sha256"])
    plan_sha = config["train_index_sha256"]
    existing = list(completed_teacher_shards(root, plan_sha, config_sha))
    terminal = root / "completion.json"
    if terminal.exists():
        return validated_teacher_receipt(root, plan_sha, config_sha)
    examples = rollout_examples(index, config["raw_shards"])
    skipped = sum(part["rows"] for part in existing)
    examples = itertools.islice(examples, skipped, None)
    head = target.model.get_output_embeddings()
    part_count = len(existing)
    counts = skipped
    for part_index in itertools.count(part_count):
        batch = list(itertools.islice(examples, config["teacher_shard_rows"]))
        if not batch:
            break
        values = {}
        for offset, row in enumerate(batch):
            with torch.inference_mode():
                output, states = target._forward_with_final_hidden(target._ids(row["prefix_ids"]))
                hidden = states[0, -1].contiguous().detach()
                native = output.logits[0, -1]
                rebuilt = head(hidden.reshape(1, -1))[0]
                require(torch.equal(native, rebuilt) and bool(native.isfinite().all()),
                        f"Target reconstruction mismatch at {row['prefix_sha256']}")
                values[str(offset)] = hidden.cpu()
                row["target_greedy_id"] = int(native.argmax())
                row["hidden_key"] = str(offset)
        stem = f"part-{part_index:08d}"
        rows = root / f"{stem}.jsonl"
        values_path = root / f"{stem}.safetensors"
        tmp_tensor = values_path.with_suffix(".safetensors.pending")
        tmp_rows = rows.with_suffix(".jsonl.pending")
        save_file(values, str(tmp_tensor))
        tmp_rows.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in batch),
                            encoding="utf-8")
        tmp_rows.replace(rows)
        tmp_tensor.replace(values_path)
        atomic_json(root / f"{stem}.json", {
            "part": part_index, "rows": len(batch), "plan_sha256": plan_sha,
            "config_sha256": config_sha, "rows_path": rows.name, "rows_sha256": digest_file(rows),
            "hidden_path": values_path.name, "hidden_sha256": digest_file(values_path),
        })
        counts += len(batch)
        part_count += 1
        atomic_json(root / "status.json", {"phase": "collect", "sealed_rows": counts, "parts": part_count})
    require(counts == config["expected_teacher_rows"], "training rollout coverage did not match locked count")
    receipt = {"status": "complete", "rows": counts, "parts": part_count,
               "config_sha256": config_sha, "plan_sha256": plan_sha}
    atomic_json(terminal, receipt)
    return receipt


def validated_teacher_receipt(root, plan_sha, config_sha):
    root = Path(root)
    receipt = read_json(root / "completion.json")
    shards = list(completed_teacher_shards(root, plan_sha, config_sha))
    require(receipt["status"] == "complete" and receipt["config_sha256"] == config_sha
            and receipt["plan_sha256"] == plan_sha and receipt["parts"] == len(shards)
            and receipt["rows"] == sum(row["rows"] for row in shards), "teacher terminal differs")
    return receipt


def teacher_rows(root, config, config_sha):
    from safetensors import safe_open
    root = Path(root)
    plan_sha = config["train_index_sha256"]
    for part in completed_teacher_shards(root, plan_sha, config_sha):
        with safe_open(root / part["hidden_path"], framework="pt", device="cpu") as tensors:
            for row in jsonl(root / part["rows_path"]):
                source = row["previous_feedback"]
                if source is not None:
                    rid = source["request_id"]
                    require(rid == row["request_id"] and Path(rid).name == rid,
                            "teacher cache feedback request differs")
                    source["path"] = str(Path(config["raw_shards"]) / rid / "features.safetensors")
                yield row, tensors.get_tensor(row["hidden_key"])
