"""Evaluate the completed 478-part pilot on the original full503 development set.

Training config/checkpoint remain unchanged. Historical baselines are used only
for acceptance counts, never as fresh timing measurements.
"""
import argparse
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path

from program62_data import atomic_json, checked_file, digest_file, jsonl, read_json, require
from program62_compare import compare_by_family

ROSTER_SHA = "a75803ceae25fcf21a4ae42cb87470bca375096d9e0676ed52645a727ae4a836"
ROSTER_CONTENT_SHA = "a06a036b827f582014765a8f11b8253ab93f61cc09a2f157f8e0b537d7f58e04"
USER_MESSAGE = "49a8f371"


def sealed(path, value):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, f"existing evaluation identity differs: {path}")
    else:
        atomic_json(path, value)


def validate_training(cfg, cfg_sha, terminal, payload):
    subset = cfg["pilot_subset"]
    require(subset["parts"] == 478 and subset["full_original_collection_complete"] is False,
            "not the user-authorized partial pilot")
    require(cfg["max_updates"] is None, "not the complete one-pass pilot budget")
    rows = subset["eligible_feedback_rows"]
    updates = math.ceil(rows / cfg["tokens_per_step"])
    require(rows > 0 and terminal["status"] == "complete", "training is not complete")
    for record in (terminal, payload):
        require(record["arm"] == "kv" and record["config_sha256"] == cfg_sha
                and record["next_row"] == rows and record["update"] == updates,
                "checkpoint is not the completed same-config pilot")
        require(record["teacher_sha256"] == terminal["teacher_sha256"],
                "training teacher binding differs")
    require(terminal["parameter_count"] == payload["parameter_count"] == 8388608,
            "single-slot parameter count differs")


def validate_roster(path, cfg):
    checked_file(path, ROSTER_SHA)
    rows = list(jsonl(path))
    require(len(rows) == 503 and len({r["request_id"] for r in rows}) == 503
            and len({r["family_id"] for r in rows}) == 503, "not original full503")
    checked_file(cfg["train_index"], cfg["train_index_sha256"])
    train_families = {r["family_id"] for r in jsonl(cfg["train_index"]) if r["split"] == "train"}
    require(train_families.isdisjoint(r["family_id"] for r in rows), "evaluation family overlaps training")
    require((cfg["max_new"], cfg["window"], cfg["eos_token_id"]) == (128, 8, 151645),
            "historical decoding horizon differs")
    for row in rows:
        require(row["prompt_token_ids"] and 0 < len(row["expected_target_token_ids"]) <= 128,
                "invalid original request")
        for counts in row["baseline_counts"].values():
            require(len(counts) == 2 and counts[0] >= 0 and counts[1] > 0,
                    "invalid historical count baseline")
    return rows


def validate_row(record, request, cfg, mode):
    require(record["request_id"] == request["request_id"] and record["family_id"] == request["family_id"]
            and record["mode"] == mode, "sealed request identity differs")
    expected = request["expected_target_token_ids"]
    prefix = request["prompt_token_ids"].copy()
    cursor = accepted_total = 0
    rounds = record["rounds"]
    require(rounds and record["verification_rounds"] == len(rounds), "invalid round total")
    for number, row in enumerate(rounds):
        proposal = row["candidate_ids"]
        require(row["round"] == number and row["prefix_before_ids"] == prefix,
                "round prefix/ordering differs")
        feedback = row["feedback_from_round"]
        require(feedback is None if number == 0 or mode == "mask"
                else feedback == number - 1 if mode == "true" else feedback in (None, number - 1),
                "feedback is not from the previous legal round")
        if mode == "shuffle" and feedback is not None:
            donor = row["feedback_donor"]
            require(donor["request_id"] != request["request_id"]
                    and donor["family_id"] != request["family_id"], "shuffle donor leaks evaluation family")
        require(proposal and len(proposal) <= min(cfg["window"], cfg["max_new"] - cursor)
                and cfg["eos_token_id"] not in proposal[:-1], "invalid proposal horizon")
        accepted = 0
        for token in proposal:
            if cursor + accepted >= len(expected) or token != expected[cursor + accepted]:
                break
            accepted += 1
            if token == cfg["eos_token_id"]:
                break
        require(accepted == row["accepted_count"], "acceptance does not match actual proposal")
        terminal = accepted > 0 and (proposal[accepted - 1] == cfg["eos_token_id"]
                                     or cursor + accepted == cfg["max_new"])
        committed = row["committed_ids"]
        require(len(committed) == accepted + int(accepted < len(proposal) or not terminal)
                and committed == expected[cursor:cursor + len(committed)]
                and row["target_decision_ids"] == committed, "Target decision/commit differs")
        prefix += committed
        require(row["prefix_after_ids"] == prefix and row["confirmed_prefix_matches_frozen_oracle"] is True,
                "sealed confirmed prefix differs")
        cursor += len(committed)
        accepted_total += accepted
    require(cursor == len(expected) and record["output_ids"] == expected
            and accepted_total == record["accepted_draft_tokens"]
            and record["actual_target_forwards"] == len(expected), "request totals/output differ")
    require(math.isfinite(record["wall_seconds"]) and math.isfinite(record["first_commit_seconds"])
            and 0 < record["first_commit_seconds"] <= record["wall_seconds"],
            "invalid request timing")


def run_rows(runtime, cfg, rows, binding, payload, target, draft, out):
    content_sha = sha256((json.dumps(rows, sort_keys=True) + "\n").encode()).hexdigest()
    require(len(rows) == 503 and binding["roster_sha256"] == ROSTER_SHA
            and binding["ordered_roster_content_sha256"] == content_sha == ROSTER_CONTENT_SHA,
            "evaluation roster binding differs")
    out = Path(out)
    sealed(out / "binding.json", binding)
    sealed(out / "roster.json", rows)
    adapter = runtime.make_adapter("kv", draft, target, cfg)
    adapter.load_state_dict(payload["adapter"], strict=True)
    draft.model.requires_grad_(False).eval()
    donors = None
    if binding["mode"] == "shuffle":
        donors = runtime.Donors(runtime.rollout_examples(cfg["train_index"], cfg["raw_shards"]),
                                {r["family_id"] for r in rows})
    decode = runtime.decoder(cfg)
    runtime.request_eval(cfg, decode, target, draft, rows[0], adapter, binding["mode"], donors)
    results = {}
    wall = 0.0
    for i, request in enumerate(rows):
        path = out / "rows" / (sha256(request["request_id"].encode()).hexdigest() + ".json")
        if path.exists():
            record = read_json(path)
            require(record["binding"] == binding, "existing row belongs to another evaluation")
        else:
            record = runtime.request_eval(cfg, decode, target, draft, request, adapter, binding["mode"], donors)
            validate_row(record, request, cfg, binding["mode"])
            atomic_json(path, {**record, "binding": binding})
        validate_row(record, request, cfg, binding["mode"])
        results[request["request_id"]] = (record["accepted_draft_tokens"], record["verification_rounds"])
        wall += record["wall_seconds"]
        atomic_json(out / "status.json", dict(sealed_requests=i + 1, total=503,
                    measured_request_seconds=wall, last_request_seconds=record["wall_seconds"], **binding))
    roster = {r["request_id"]: r for r in rows}
    controls = ("RAW", "mlp1_w256_" + cfg["weighting"], "transformer1_w64_" + cfg["weighting"])
    comparisons = {name: compare_by_family(roster,
        {rid: tuple(row["baseline_counts"][name]) for rid, row in roster.items()}, results, cfg["seed"], 10000)
        for name in controls}
    accepted = sum(v[0] for v in results.values())
    rounds = sum(v[1] for v in results.values())
    result = dict(status="complete", requests=len(results), accepted_draft_tokens=accepted,
                  verification_rounds=rounds, accepted_per_round=accepted / rounds,
                  measured_request_seconds=wall, comparisons=comparisons, **binding)
    atomic_json(out / "comparison.json", result)
    atomic_json(out / "completion.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, default=Path(__file__).resolve().parent / "inputs/development503_existing.jsonl")
    parser.add_argument("--mode", choices=("true", "mask", "shuffle"), default="true")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parent
    root = args.training_root.resolve()
    out = root / "eval503" / args.mode
    require(out.resolve() == out and not root.is_relative_to(checkout),
            "evaluation output aliases source or another directory")
    cfg = read_json(root / "config.json")
    cfg_sha = digest_file(root / "config.json")
    admission = read_json(root / "pilot_admission.json")
    require(admission["pilot_config_sha256"] == cfg_sha, "pilot admission/config differs")
    checked_file(checkout / "pilot478.py", admission["helper_sha256"])
    for name, expected in cfg["implementation_sha256"].items():
        checked_file(checkout / name, expected)
    rows = validate_roster(args.roster, cfg)
    runtime = importlib.import_module("program62_run")
    require(Path(runtime.__file__).resolve() == checkout / "program62_run.py", "runtime import differs")
    terminal_path = root / "kv/completion.json"
    terminal = read_json(terminal_path)
    ckpt = root / "kv/last.pt"
    checked_file(ckpt, terminal["checkpoint_sha256"])
    payload = runtime.torch.load(ckpt, map_location="cpu", weights_only=True)
    validate_training(cfg, cfg_sha, terminal, payload)
    checked_file(Path(cfg["teacher_root"]) / "completion.json", terminal["teacher_sha256"])
    binding = dict(arm="kv", mode=args.mode, split="original503-development",
        training_config_sha256=cfg_sha, training_completion_sha256=digest_file(terminal_path),
        checkpoint_sha256=terminal["checkpoint_sha256"], roster_sha256=ROSTER_SHA,
        ordered_roster_content_sha256=sha256((json.dumps(rows, sort_keys=True) + "\n").encode()).hexdigest(),
        evaluator_sha256=digest_file(__file__), pilot_admission_sha256=digest_file(root / "pilot_admission.json"),
        user_message=USER_MESSAGE, warmup_requests=1,
        scope="Original complete503; partial478 one-pass pilot; historical count baselines only. "
              "No fresh RAW timing/speed claim; no independent confirmation; training budgets differ.")
    runtime.torch.manual_seed(cfg["seed"])
    with runtime.worker_lease(cfg, out):
        checked_file(ckpt, terminal["checkpoint_sha256"])
        target, draft = runtime.load_models(cfg)
        checked_file(ckpt, terminal["checkpoint_sha256"])
        run_rows(runtime, cfg, rows, binding, payload, target, draft, out)
    checked_file(root / "config.json", cfg_sha)
    checked_file(ckpt, terminal["checkpoint_sha256"])
    print("complete full503", args.mode, terminal["checkpoint_sha256"])


if __name__ == "__main__":
    main()
