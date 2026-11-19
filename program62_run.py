"""Collect, train, preflight, or evaluate the two independent Task62 arms."""

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import time

import torch

from draft_interventions import Feedback, SingleSlot, attach_lora, enable_lora
from program62_data import (
    atomic_json, checked_file, collect_teacher, digest_file, feedback_hidden,
    jsonl, read_json, require, rollout_examples, teacher_rows, validated_teacher_receipt,
    exact_logit_payload, rebuild_teacher_logits,
)
from rollout_coverage import prefix_sha


def load_models(cfg, collect_only=False):
    checked_file(cfg["model_manifest"], cfg["model_manifest_sha256"])
    manifest = read_json(cfg["model_manifest"])
    root = Path(cfg["models_root"])
    names = ("Qwen3-8B",) if collect_only else ("Qwen3-8B", "Qwen3-0.6B")
    for name in names:
        for file, entry in manifest[name].items():
            checked_file(root / name / file, entry["sha256"])
    sys.path.insert(0, cfg["source_root"])
    checked_file(Path(cfg["source_root"]) / "feedback_codec" / "hf_backend.py", cfg["backend_sha256"])
    from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
    model_args = {"device": cfg["device"], "dtype": torch.bfloat16}
    target = HFTargetBackend(str(root / "Qwen3-8B"), None, **model_args)
    if collect_only:
        return target, None
    draft = HFDraftBackend(str(root / "Qwen3-0.6B"), None, None, target=target, **model_args)
    return target, draft


def decoder(cfg):
    checked_file(cfg["decoder"], cfg["decoder_sha256"])
    spec = importlib.util.spec_from_file_location("task62_native_closed_loop", cfg["decoder"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode


def make_adapter(arm, draft, target, cfg):
    if arm == "lora":
        return attach_lora(draft.model, cfg["lora_rank"], cfg["lora_alpha"])
    return SingleSlot(draft.model, cfg["slot_layer"], target.hidden_size)


def draft_logits(draft, prefix, slot, state):
    with slot.activate(state) if slot is not None else nullcontext():
        decoder_output = draft.model.model(
            input_ids=draft._ids(prefix), use_cache=False, return_dict=True,
        )
        hidden = decoder_output.last_hidden_state[0, -1:]
        return draft.model.get_output_embeddings()(hidden)[0].float()


def token_weight(depth, weighting):
    require(weighting in ("equal", "front") and 1 <= depth <= 8, "invalid loss weight or draft depth")
    return 1 if weighting == "equal" else 9 - depth


def equal_token_step(rows, draft, target, adapter, slot, optimizer, weighting="equal"):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    weights = [token_weight(row.get("draft_depth", 1), weighting) for row, _ in rows]
    total_weight = sum(weights)
    for (row, teacher_hidden), weight in zip(rows, weights):
        require(prefix_sha(row["prefix_ids"]) == row["prefix_sha256"], "training prefix changed")
        source = row["previous_feedback"]
        state = Feedback(feedback_hidden(source), row["visible_from"]) if slot and source else None
        require(slot is None or state is not None, "training slot received a first-round row")
        with torch.no_grad():
            label_logits = rebuild_teacher_logits(row, teacher_hidden, target)
            require(int(label_logits.argmax()) == row["target_greedy_id"], "teacher greedy label changed")
            log_p = label_logits.log_softmax(-1)
        log_q = draft_logits(draft, row["prefix_ids"], slot, state).log_softmax(-1)
        loss = torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction="sum")
        require(bool(loss.isfinite()), "nonfinite FKL")
        (loss * (weight / total_weight)).backward()
        losses.append(float(loss.detach()) * weight)
    if losses:
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return sum(losses) / total_weight, float(norm)
    return 0., 0.


def train(cfg, cfg_sha, arm, target, draft, root, teacher_root, teacher_sha_config):
    teacher = validated_teacher_receipt(teacher_root, cfg["train_index_sha256"], teacher_sha_config)
    teacher_sha = digest_file(teacher_root / "completion.json")
    adapter = make_adapter(arm, draft, target, cfg)
    slot = adapter if arm == "kv" else None
    parameters = dict(adapter.named_parameters())
    require(parameters and all(param.requires_grad for param in parameters.values()), "invalid adapter parameters")
    optimizer = torch.optim.AdamW(parameters.values(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    resume = root / "last.pt"
    state = {"config_sha256": cfg_sha, "teacher_sha256": teacher_sha, "arm": arm,
             "next_row": 0, "update": 0, "parameter_count": sum(p.numel() for p in parameters.values())}
    if resume.exists():
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        require(all(payload[key] == state[key] for key in ("config_sha256", "teacher_sha256", "arm", "parameter_count")),
                "adapter training resume contract differs")
        adapter.load_state_dict(payload["adapter"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(next_row=payload["next_row"], update=payload["update"])
    batch_size = cfg["tokens_per_step"]
    limit = cfg["max_updates"]
    require(batch_size > 0 and teacher["rows"] > 0 and (limit is None or limit > 0),
            "empty training budget")
    # Exclude first-round rows in both training arms to keep identical FKL weights.
    valid_rows = ((row, hidden) for row, hidden in teacher_rows(teacher_root, cfg, teacher_sha_config)
                  if row["previous_feedback"] is not None)
    data = islice(valid_rows, state["next_row"], None)
    while limit is None or state["update"] < limit:
        batch = list(islice(data, batch_size))
        if not batch:
            break
        loss, norm = equal_token_step(batch, draft, target, adapter, slot, optimizer, cfg["weighting"])
        state["next_row"] += len(batch)
        state["update"] += 1
        if state["update"] % cfg["checkpoint_interval"] == 0 or state["update"] == 1:
            checkpoint(resume, state, adapter, optimizer)
            atomic_json(root / "status.json", {**state, "last_fkl": loss, "gradient_norm": norm})
    require(state["update"] > 0 and (limit is None or state["update"] == limit),
            "locked update budget exceeds available rollout data")
    checkpoint(resume, state, adapter, optimizer)
    atomic_json(root / "completion.json", {**state, "status": "complete", "checkpoint_sha256": digest_file(resume)})


def checkpoint(path, state, adapter, optimizer):
    tmp = path.with_suffix(".pt.pending")
    torch.save({**state, "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
    tmp.replace(path)


class Donors:
    def __init__(self, examples, development_families):
        self.by_depth = defaultdict(list)
        seen = set()
        for row in examples:
            source = row["previous_feedback"]
            if source is None:
                continue
            require(row["family_id"] not in development_families, "donor collection leaks development family")
            key = (row["request_id"], row["round_id"])
            if key in seen:
                continue
            seen.add(key)
            self.by_depth[(source["accepted"], source["generated"] // 16)].append(
                (row["family_id"], row["request_id"], row["round_id"], source)
            )

    def hidden(self, request, row):
        options = self.by_depth.get((row["accepted_count"], row["output_tokens_so_far"] // 16), ())
        if not options:
            return None
        key = f"{request['request_id']}:{row['round']}".encode()
        choice = int.from_bytes(sha256(key).digest()[:8], "big") % len(options)
        family, request_id, round_id, source = options[choice]
        require(family != request["family_id"], "SHUFFLE donor shares eval family")
        return feedback_hidden(source), {"request_id": request_id, "round_id": round_id, "family_id": family}


def request_eval(cfg, decode, target, draft, request, slot=None, mode="raw", donors=None):
    feedback_donors = {}
    first_token_at = None

    def token(prefix, state, position):
        with slot.activate(state) if slot is not None else nullcontext():
            return draft.next(prefix, None, position)

    def target_step(prefix):
        step = target.next(prefix)
        return SimpleNamespace(token_id=step.token_id, hidden=step.hidden, source_prefix=tuple(prefix))

    def prepare(row, boundary):
        require(tuple(boundary.source_prefix) == tuple(row["prefix_after_ids"][:-1]),
                "online feedback is not from this round's legal Target boundary")
        if mode == "mask":
            return None
        hidden = boundary.hidden
        if mode == "shuffle":
            choice = donors.hidden(request, row)
            if choice is None:
                return None
            hidden, feedback_donors[row["round"] + 1] = choice
        return None if hidden is None else Feedback(hidden.detach(), len(row["prefix_after_ids"]) - 1)

    def record_round(row, boundary):
        nonlocal first_token_at
        if first_token_at is None:
            if draft.device.type == "cuda":
                torch.cuda.synchronize(draft.device)
            first_token_at = time.perf_counter()
        if row["round"] in feedback_donors:
            row["feedback_donor"] = feedback_donors[row["round"]]

    if draft.device.type == "cuda":
        torch.cuda.synchronize(draft.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = decode(
            request["prompt_token_ids"], draft_token=token, target_step=target_step,
            prepare=prepare if slot is not None else None, on_round=record_round,
            oracle_ids=request["expected_target_token_ids"], max_new=cfg["max_new"], window=cfg["window"],
            eos=cfg["eos_token_id"],
        )
    if draft.device.type == "cuda":
        torch.cuda.synchronize(draft.device)
    result["wall_seconds"] = time.perf_counter() - started
    result["first_commit_seconds"] = first_token_at - started
    result["request_id"] = request["request_id"]
    result["family_id"] = request["family_id"]
    result["mode"] = mode
    return result


def decision_signature(result):
    keys = ("candidate_ids", "target_decision_ids", "accepted_count", "committed_ids", "eos", "truncated")
    return result["output_ids"], [[row[key] for key in keys] for row in result["rounds"]]


def smoke(cfg, cfg_sha, arm, target, draft, root):
    checked_file(cfg["train_index"], cfg["train_index_sha256"])
    checked_file(cfg["development_roster"], cfg["development_roster_sha256"])
    decode = decoder(cfg)
    adapter = make_adapter(arm, draft, target, cfg)
    slot = adapter if arm == "kv" else None
    examples = rollout_examples(cfg["train_index"], cfg["raw_shards"])
    rows = []
    for row in islice((r for r in examples if r["previous_feedback"]), 8):
        with torch.inference_mode():
            output, hidden = target._forward_with_final_hidden(target._ids(row["prefix_ids"]))
            last = hidden[0, -1].detach().cpu()
            expected = output.logits[0, -1]
            rebuilt = target.model.get_output_embeddings()(last.to(target.device).reshape(1, -1))[0]
            row["_teacher_logit_patch"] = exact_logit_payload(expected, rebuilt)
            require(torch.equal(rebuild_teacher_logits(row, last, target), expected.float()),
                    "smoke native teacher codec failed")
            row["target_greedy_id"] = int(expected.argmax())
        rows.append((row, last.clone()))
    require(len(rows) == 8, "insufficient legal training smoke rows")
    parity = list(islice(jsonl(cfg["development_roster"]), 3))
    require(len(parity) == 3, "insufficient development parity rows")
    for request in parity:
        if arm == "lora":
            enable_lora(draft.model, False)
        baseline = request_eval(cfg, decode, target, draft, request)
        if arm == "lora":
            enable_lora(draft.model, True)
        bypass = request_eval(cfg, decode, target, draft, request, slot, "mask")
        require(decision_signature(baseline) == decision_signature(bypass), "native RAW parity differs")
    frozen = {name: param._version for name, param in draft.model.named_parameters() if not param.requires_grad}
    previous = [param.detach().clone() for param in adapter.parameters()]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"])
    loss, norm = equal_token_step(rows, draft, target, adapter, slot, optimizer, cfg["weighting"])
    require(norm > 0 and any(not torch.equal(old, new) for old, new in zip(previous, adapter.parameters())),
            "smoke did not update the adapter")
    require(all(param._version == frozen[name] and param.grad is None
                for name, param in draft.model.named_parameters() if name in frozen),
            "frozen Draft changed during smoke")
    require(all(param.grad is None for param in target.model.parameters()), "Target accumulated a gradient")
    atomic_json(root / "smoke.json", {"status": "pass", "arm": arm, "config_sha256": cfg_sha,
                "checked_development_requests": 3, "training_rows": 8, "FKL": loss, "gradient_norm": norm,
                "native_teacher_codec": [
                    {key: value for key, value in row["_teacher_logit_patch"].items() if key not in ("ids", "values")}
                    for row, _ in rows]})


@contextmanager
def worker_lease(cfg, root):
    from os import getpid
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    path = cfg["gpu_lock"] if cfg["device"].startswith("cuda") else str(root / "cpu.lock")
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {"pid": getpid(), "device": cfg["device"], "start_unix": time.time(), "lock": path}
        atomic_json(root / "gpu_lease.json", {**receipt, "status": "held"})
        try:
            yield
        finally:
            atomic_json(root / "gpu_lease.json", {**receipt, "status": "released", "end_unix": time.time()})


@contextmanager
def teacher_writer(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def evaluate(cfg, cfg_sha, arm, mode, target, draft, root, ckpt, split):
    path = cfg[f"{split}_roster"]
    checked_file(path, cfg[f"{split}_roster_sha256"])
    requests = list(jsonl(path))
    families = {r["family_id"] for r in requests}
    require(len(requests) == len({row["request_id"] for row in requests}) and requests, "invalid eval roster")
    if split == "confirmation":
        checked_file(cfg["development_roster"], cfg["development_roster_sha256"])
        checked_file(cfg["train_index"], cfg["train_index_sha256"])
        old_families = {row["family_id"] for row in jsonl(cfg["development_roster"])}
        old_families.update(row["family_id"] for row in jsonl(cfg["train_index"]) if row["split"] == "train")
        require(len(families) == len(requests) and families.isdisjoint(old_families),
                "confirmation family overlap")
    decode = decoder(cfg)
    if mode == "parity":
        adapter = make_adapter(arm, draft, target, cfg)
        slot = adapter if arm == "kv" else None
        for request in requests:
            if arm == "lora":
                enable_lora(draft.model, False)
            baseline = request_eval(cfg, decode, target, draft, request)
            if arm == "lora":
                enable_lora(draft.model, True)
            candidate = request_eval(cfg, decode, target, draft, request, slot=slot, mode="mask")
            require(decision_signature(baseline) == decision_signature(candidate),
                    f"zero adapter or MASK differs from RAW: {request['request_id']}")
        atomic_json(root / "completion.json", {"status": "pass", "arm": arm, "requests": len(requests),
                                                "config_sha256": cfg_sha})
        return
    slot = None
    donors = None
    if mode != "raw":
        adapter = make_adapter(arm, draft, target, cfg)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        require(payload["arm"] == arm and payload["config_sha256"] == cfg_sha, "evaluation checkpoint differs")
        adapter.load_state_dict(payload["adapter"])
        if arm == "kv":
            slot = adapter
            if mode == "shuffle":
                checked_file(cfg["train_index"], cfg["train_index_sha256"])
                donors = Donors(rollout_examples(cfg["train_index"], cfg["raw_shards"]), families)
    binding = {"config_sha256": cfg_sha, "arm": arm, "mode": mode, "split": split,
               "checkpoint_sha256": digest_file(ckpt) if mode != "raw" else None,
               "warmup_requests": cfg.get("warmup_requests", 1)}
    sums = defaultdict(float)
    for request in requests[:binding["warmup_requests"]]:
        request_eval(cfg, decode, target, draft, request, slot, mode, donors)
    for i, request in enumerate(requests):
        sealed = root / "rows" / (sha256(request["request_id"].encode()).hexdigest() + ".json")
        if sealed.exists():
            record = read_json(sealed)
            require(record["binding"] == binding and record["request_id"] == request["request_id"]
                    and record["output_ids"] == request["expected_target_token_ids"], "sealed eval request differs")
        else:
            record = request_eval(cfg, decode, target, draft, request, slot, mode, donors)
            atomic_json(sealed, {**record, "binding": binding})
        for key in ("accepted_draft_tokens", "verification_rounds", "wall_seconds"):
            sums[key] += record[key]
        sums["output_tokens"] += len(record["output_ids"])
        sums["feedback_rounds"] += sum(row["feedback_from_round"] is not None for row in record["rounds"])
        atomic_json(root / "status.json", {"sealed_requests": i + 1, "total": len(requests), **binding})
    atomic_json(root / "completion.json", {**binding, **sums, "requests": len(requests), "status": "complete",
                "accepted_per_round": sums["accepted_draft_tokens"] / sums["verification_rounds"],
                "tokens_per_second": sums["output_tokens"] / sums["wall_seconds"]})


def fixed_windows(cfg, cfg_sha, arm, mode, target, draft, root, arm_root):
    checked_file(cfg["development_roster"], cfg["development_roster_sha256"])
    roster = list(jsonl(cfg["development_roster"]))
    slot = None
    donors = None
    checkpoint_sha = None
    if mode != "raw":
        adapter = make_adapter(arm, draft, target, cfg)
        path = arm_root / "last.pt"
        terminal = read_json(arm_root / "completion.json")
        require(terminal["config_sha256"] == cfg_sha and terminal["status"] == "complete",
                "fixed-window checkpoint binding differs")
        checked_file(path, terminal["checkpoint_sha256"])
        checkpoint_sha = terminal["checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["arm"] == arm and payload["config_sha256"] == cfg_sha, "wrong fixed-window checkpoint")
        adapter.load_state_dict(payload["adapter"])
        slot = adapter if arm == "kv" else None
        if mode == "shuffle":
            checked_file(cfg["train_index"], cfg["train_index_sha256"])
            donors = Donors(rollout_examples(cfg["train_index"], cfg["raw_shards"]),
                            {request["family_id"] for request in roster})
    windows = []
    with torch.inference_mode():
        for request in roster:
            key = sha256(request["request_id"].encode()).hexdigest()
            raw = read_json(arm_root / "development-raw" / "rows" / (key + ".json"))
            require(raw["binding"]["config_sha256"] == cfg_sha
                    and raw["output_ids"] == request["expected_target_token_ids"], "fixed anchor RAW differs")
            if len(raw["rounds"]) < 2:
                continue
            # First round after feedback, the prespecified common anchor.
            previous, start = raw["rounds"][:2]
            prefix = start["prefix_before_ids"]
            offset = len(prefix) - len(request["prompt_token_ids"])
            labels = raw["output_ids"][offset:offset + cfg["window"]]
            state = None
            if slot is not None and mode == "true":
                step = target.next(prefix[:-1])
                require(step.token_id == prefix[-1], "fixed-window Target boundary differs")
                state = Feedback(step.hidden, len(prefix) - 1)
            if slot is not None and mode == "shuffle":
                donor = donors.hidden(request, previous)
                if donor:
                    state = Feedback(donor[0], len(prefix) - 1)
            proposal = []
            accepted = 0
            for position, label in enumerate(labels, start=1):
                with slot.activate(state) if slot is not None else nullcontext():
                    greedy = draft.next(prefix + proposal, None, position)
                proposal.append(greedy)
                if greedy == label and accepted == len(proposal) - 1:
                    accepted += 1
                if greedy == cfg["eos_token_id"]:
                    break
            windows.append({"request_id": request["request_id"], "family_id": request["family_id"],
                            "prefix_sha256": prefix_sha(prefix), "target_ids": labels, "candidate_ids": proposal,
                            "accepted": accepted, "feedback_present": state is not None,
                            "feedback_donor": donor[1] if mode == "shuffle" and state else None})
    require(windows, "no common feedback anchor")
    atomic_json(root / "completion.json", {"status": "complete", "config_sha256": cfg_sha, "mode": mode,
                "checkpoint_sha256": checkpoint_sha,
                "windows": len(windows), "total_accepted": sum(row["accepted"] for row in windows),
                "feedback_windows": sum(row["feedback_present"] for row in windows), "rows": windows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "collect", "train", "eval", "fixed"), required=True)
    parser.add_argument("--arm", choices=("lora", "kv"), default="lora")
    parser.add_argument("--mode", choices=("raw", "lora", "true", "mask", "shuffle", "parity"), default="parity")
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    args = parser.parse_args()
    cfg, cfg_sha = read_json(args.config), digest_file(args.config)
    for name, expected in cfg["implementation_sha256"].items():
        checked_file(Path(__file__).parent / name, expected)
    require(args.phase not in ("eval", "fixed") or args.mode in ({"raw", "lora", "parity"} if args.arm == "lora"
                                                  else {"raw", "true", "mask", "shuffle", "parity"}),
            "evaluation mode does not match its trained arm")
    torch.manual_seed(cfg["seed"])
    teacher = Path(cfg["teacher_root"])
    teacher_fields = ("backend_sha256", "model_manifest_sha256", "train_index_sha256", "expected_teacher_rows",
                      "teacher_shard_rows", "device")
    teacher_source = {key: cfg[key] for key in teacher_fields}
    for path in ("program62_data.py", "rollout_coverage.py"):
        teacher_source[path] = cfg["implementation_sha256"][path]
    teacher_sha_config = sha256(json.dumps(teacher_source, sort_keys=True).encode()).hexdigest()
    arm_path = args.root / args.arm
    with worker_lease(cfg, args.root), (teacher_writer(teacher) if args.phase == "collect" else nullcontext()):
        target, draft = load_models(cfg, collect_only=args.phase == "collect")
        arm_path.mkdir(parents=True, exist_ok=True)
        if args.phase == "smoke":
            smoke(cfg, cfg_sha, args.arm, target, draft, arm_path)
        elif args.phase == "collect":
            collect_teacher(cfg, teacher_sha_config, target, teacher)
        elif args.phase == "train":
            preflight = read_json(arm_path / "smoke.json")
            require(preflight["status"] == "pass" and preflight["config_sha256"] == cfg_sha
                    and preflight["arm"] == args.arm, "same-arm smoke not sealed")
            train(cfg, cfg_sha, args.arm, target, draft, arm_path, teacher, teacher_sha_config)
        elif args.phase == "fixed":
            require(args.split == "development" and args.mode != "parity", "invalid fixed-window phase")
            fixed_windows(cfg, cfg_sha, args.arm, args.mode, target, draft, arm_path / f"fixed-{args.mode}", arm_path)
        else:
            if args.mode not in ("raw", "parity"):
                checkpoint_path = arm_path / "last.pt"
                terminal = read_json(arm_path / "completion.json")
                require(terminal["status"] == "complete" and terminal["config_sha256"] == cfg_sha,
                        "evaluation training terminal differs")
                checked_file(checkpoint_path, terminal["checkpoint_sha256"])
            result = arm_path / f"{args.split}-{args.mode}"
            evaluate(cfg, cfg_sha, args.arm, args.mode, target, draft, result, arm_path / "last.pt", args.split)


if __name__ == "__main__":
    main()
