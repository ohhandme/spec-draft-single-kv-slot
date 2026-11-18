"""Audit FKL coverage on the exact prefixes proposed by the native Draft."""

import argparse
from collections import Counter
import hashlib
import json


def prefix_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def draft_rows(trace):
    """Yield each scored Draft position, including the exact first-reject history."""
    request_id = trace["request_id"]
    if trace["repair_enabled"] is not False or trace["trajectory_policy"] != "RAW":
        raise ValueError(f"{request_id}: missing native RAW trace")
    if trace["independent_greedy_checked"] and trace["output_ids"] != trace["oracle_ids"]:
        raise ValueError(f"{request_id}: RAW and independent Target outputs differ")
    prefix = trace["prompt_ids"].copy()
    output = []
    previous_observation = None
    for index, round_row in enumerate(trace["rounds"]):
        proposal = round_row["candidate_ids"]
        decisions = round_row["target_decision_ids"]
        accepted = round_row["accepted_count"]
        reject_at = round_row["first_reject_position"]
        if (
            round_row["request_id"] != request_id
            or round_row["round_id"] != index
            or round_row["prefix_before_sha256"] != prefix_sha(prefix)
            or not proposal
            or not 0 <= accepted <= len(proposal)
            or decisions[:accepted] != proposal[:accepted]
            or reject_at != (accepted + 1 if accepted < len(proposal) else None)
            or len(decisions) not in (accepted, accepted + 1)
        ):
            raise ValueError(f"{request_id}: invalid RAW round {index}")
        if reject_at is not None and (
            len(decisions) <= accepted or decisions[accepted] == proposal[accepted]
        ):
            raise ValueError(f"{request_id}: invalid first rejection in round {index}")
        committed = round_row["committed_ids"]
        if not committed or decisions != committed:
            raise ValueError(f"{request_id}: Target decisions do not bind to commit in round {index}")
        for position, token in enumerate(proposal):
            if position < accepted:
                kind = "accepted"
            elif position == accepted:
                kind = "first_reject"
            else:
                kind = "after_rejected_token"
            yield {
                "request_id": request_id,
                "round_id": index,
                "absolute_prediction_position": len(prefix) + position,
                "prefix_sha256": prefix_sha(prefix + proposal[:position]),
                "candidate_id": token,
                "kind": kind,
                "previous_observation_key": previous_observation,
            }
        prefix.extend(committed)
        if round_row["prefix_after_commit_sha256"] != prefix_sha(prefix):
            raise ValueError(f"{request_id}: invalid prefix after commit in round {index}")
        output.extend(committed)
        previous_observation = round_row["observation_key"]
    if output != trace["output_ids"]:
        raise ValueError(f"{request_id}: RAW output differs from verified commits")


def read_jsonl(path):
    with open(path, encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                yield json.loads(line)


def audit_coverage(traces, labels, split):
    for row in labels:
        if row["split"] != split:
            raise ValueError(f"cache shard includes a row outside the {split} split")
    covered = {row["prefix_sha256"] for row in labels}
    report = {kind: Counter(total=0, covered=0) for kind in
              ("accepted", "first_reject", "after_rejected_token")}
    seen_requests = set()
    missing_unique = set()
    for trace in traces:
        request_id = trace["request_id"]
        if request_id in seen_requests:
            raise ValueError(f"duplicate request: {request_id}")
        seen_requests.add(request_id)
        for row in draft_rows(trace):
            count = report[row["kind"]]
            count["total"] += 1
            if row["prefix_sha256"] in covered:
                count["covered"] += 1
            else:
                missing_unique.add(row["prefix_sha256"])
    if not seen_requests:
        raise ValueError("empty RAW train trace")
    return {
        "split": split,
        "requests": len(seen_requests),
        "positions": report,
        "unique_missing_prefixes": len(missing_unique),
    }


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--raw-train-traces", required=True)
    args.add_argument("--verified-train-labels", required=True)
    options = args.parse_args()
    results = audit_coverage(
        read_jsonl(options.raw_train_traces),
        list(read_jsonl(options.verified_train_labels)),
        "train",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
