"""Independently compare the sealed Program62 end-to-end and fixed-window rows."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import random

from program62_data import atomic_json, checked_file, digest_file, jsonl, read_json, require


MODES = {"lora": ("raw", "lora"), "kv": ("raw", "true", "mask", "shuffle")}


def quantile(values, probability):
    values = sorted(values)
    offset = probability * (len(values) - 1)
    lower = int(offset)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (offset - lower)


def compare_by_family(roster, control, treatment, seed, samples):
    """Rows carry a numerator and denominator so each resample recomputes the ratio."""
    families = defaultdict(lambda: [0, 0, 0, 0])
    for request_id in control:
        values = families[roster[request_id]["family_id"]]
        for i, value in enumerate((*control[request_id], *treatment[request_id])):
            values[i] += value
    clusters = tuple(families.values())
    require(clusters, "comparison has no eligible requests")

    def score(rows):
        raw_tokens, raw_rounds, new_tokens, new_rounds = map(sum, zip(*rows))
        require(raw_rounds > 0 and new_rounds > 0, "comparison denominator is zero")
        return raw_tokens / raw_rounds, new_tokens / new_rounds

    raw, new = score(clusters)
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        baseline, candidate = score(rng.choices(clusters, k=len(clusters)))
        differences.append(candidate - baseline)
    return {"control": raw, "treatment": new, "delta": new - raw,
            "ci95": [quantile(differences, 0.025), quantile(differences, 0.975)],
            "requests": len(control), "families": len(clusters), "bootstrap_draws": samples}


def read_eval(root, config_sha, roster, arm, split, mode):
    completion = read_json(root / "completion.json")
    binding = {key: completion[key] for key in (
        "arm", "mode", "split", "config_sha256", "checkpoint_sha256", "warmup_requests",
    )}
    require(completion["status"] == "complete" and completion["requests"] == len(roster)
            and binding["arm"] == arm and binding["mode"] == mode and binding["split"] == split
            and binding["config_sha256"] == config_sha, f"evaluation terminal differs for {mode}")
    totals = defaultdict(int)
    rows = {}
    for request_id, expected in roster.items():
        key = sha256(request_id.encode()).hexdigest()
        row = read_json(root / "rows" / (key + ".json"))
        require(row["binding"] == binding and row["request_id"] == request_id
                and row["family_id"] == expected["family_id"]
                and row["output_ids"] == expected["expected_target_token_ids"], "evaluation row differs")
        accepted = sum(round_row["accepted_count"] for round_row in row["rounds"])
        rounds = len(row["rounds"])
        require(accepted == row["accepted_draft_tokens"] and rounds == row["verification_rounds"],
                "evaluation totals differ from round records")
        require(0 < row["first_commit_seconds"] <= row["wall_seconds"], "invalid request latency")
        totals["accepted_draft_tokens"] += accepted
        totals["verification_rounds"] += rounds
        totals["output_tokens"] += len(row["output_ids"])
        totals["wall_seconds"] += row["wall_seconds"]
        rows[request_id] = row
    for metric in ("accepted_draft_tokens", "verification_rounds", "output_tokens"):
        require(totals[metric] == completion[metric], f"evaluation terminal {metric} differs")
    require(abs(totals["wall_seconds"] - completion["wall_seconds"]) < 1e-5, "wall time totals differ")
    return rows, {**totals, "checkpoint_sha256": binding["checkpoint_sha256"],
                  "tokens_per_second": totals["output_tokens"] / totals["wall_seconds"],
                  "first_commit_p50_s": quantile([r["first_commit_seconds"] for r in rows.values()], 0.5),
                  "first_commit_p95_s": quantile([r["first_commit_seconds"] for r in rows.values()], 0.95)}


def read_fixed(root, config_sha, roster, mode, checkpoint_sha):
    terminal = read_json(root / "completion.json")
    require(terminal["mode"] == mode and terminal["config_sha256"] == config_sha
            and terminal["status"] == "complete", "fixed-window terminal differs")
    require(terminal.get("checkpoint_sha256") == checkpoint_sha, "fixed-window checkpoint differs")
    rows = {}
    for row in terminal["rows"]:
        key = row["request_id"]
        require(key in roster and key not in rows and row["family_id"] == roster[key]["family_id"],
                "fixed-window request binding differs")
        accepted = 0
        for candidate, label in zip(row["candidate_ids"], row["target_ids"]):
            if candidate != label:
                break
            accepted += 1
        require(row["accepted"] == accepted, "fixed-window continuous acceptance differs")
        rows[key] = row
    require(terminal["windows"] == len(rows) and terminal["total_accepted"] == sum(r["accepted"] for r in rows.values()),
            "fixed-window aggregate differs")
    return rows


def fixed_comparisons(rows, roster, arm, seed, samples):
    anchors = rows["raw"]
    for group in rows.values():
        require(group.keys() == anchors.keys(), "fixed-window mode coverage differs")
        for key, row in group.items():
            require(row["prefix_sha256"] == anchors[key]["prefix_sha256"]
                    and row["target_ids"] == anchors[key]["target_ids"], "fixed-window anchor differs")
    controls = (("raw", "lora"),) if arm == "lora" else (("raw", "true"), ("mask", "true"), ("shuffle", "true"))
    eligible = [key for key in anchors if arm == "lora" or rows["true"][key]["feedback_present"]]
    common_donor = [key for key in eligible if arm == "lora" or rows["shuffle"][key]["feedback_present"]]
    comparisons = {}
    for control, treatment in controls:
        keys = common_donor if control == "shuffle" else eligible
        if not keys:
            comparisons[f"{treatment}_vs_{control}"] = {"requests": 0}
            continue
        by_mode = {mode: {key: (rows[mode][key]["accepted"], 1) for key in keys}
                   for mode in (control, treatment)}
        result = compare_by_family(roster, by_mode[control], by_mode[treatment], seed, samples)
        result["first_token_spoiled"] = sum(
            rows[control][key]["accepted"] > 0 and rows[treatment][key]["accepted"] == 0 for key in keys)
        result["first_token_rescued"] = sum(
            rows[treatment][key]["accepted"] > 0 and rows[control][key]["accepted"] == 0 for key in keys)
        depth = max(len(anchors[key]["target_ids"]) for key in keys)
        result["continue_probability"] = {
            mode: [sum(rows[mode][key]["accepted"] >= d for key in keys) / len(keys)
                   for d in range(1, depth + 1)]
            for mode in (control, treatment)
        }
        comparisons[f"{treatment}_vs_{control}"] = result
    return {"anchor_requests": len(anchors), "eligible_requests": len(eligible),
            "common_donor_requests": len(common_donor), "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=MODES, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    args = parser.parse_args()
    config, config_sha = read_json(args.config), digest_file(args.config)
    roster_path = config[f"{args.split}_roster"]
    checked_file(roster_path, config[f"{args.split}_roster_sha256"])
    roster_rows = list(jsonl(roster_path))
    roster = {row["request_id"]: row for row in roster_rows}
    require(len(roster) == len(roster_rows), "duplicate evaluation request")
    root = args.root / args.arm
    modes = MODES[args.arm]
    seed, samples = 42, 10000
    observed = {mode: read_eval(root / f"{args.split}-{mode}", config_sha, roster, args.arm, args.split, mode)
                for mode in modes}
    terminal = read_json(root / "completion.json")
    require(terminal["status"] == "complete" and terminal["arm"] == args.arm and
            terminal["config_sha256"] == config_sha, "training checkpoint binding differs")
    checked_file(root / "last.pt", terminal["checkpoint_sha256"])
    for mode in modes:
        expected = None if mode == "raw" else terminal["checkpoint_sha256"]
        require(observed[mode][1]["checkpoint_sha256"] == expected, "evaluation checkpoint differs from training")
    counts = {mode: {key: (row["accepted_draft_tokens"], row["verification_rounds"]) for key, row in rows.items()}
              for mode, (rows, _) in observed.items()}
    report = {"arm": args.arm, "split": args.split, "config_sha256": config_sha,
              "totals": {mode: summary for mode, (_, summary) in observed.items()},
              "paired": {mode: compare_by_family(roster, counts["raw"], counts[mode], seed, samples)
                         for mode in modes if mode != "raw"}}
    if args.split == "development":
        fixed = {mode: read_fixed(root / f"fixed-{mode}", config_sha, roster, mode,
                                  observed[mode][1]["checkpoint_sha256"]) for mode in modes}
        report["fixed"] = fixed_comparisons(fixed, roster, args.arm, seed, samples)
    atomic_json(args.out, report)


if __name__ == "__main__":
    main()
