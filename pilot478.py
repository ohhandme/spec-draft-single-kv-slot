"""Freeze an existing 478-part subset and train it with the original single-slot math."""
import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PARTS = 478
USER_MESSAGE = "c82634c5"
CANDIDATE_SHA = "58de6751dc67a182eba769bc35d0ef784ed8701316a4a9427b13920666418113"


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == content, "existing pilot artifact differs: " + str(path))
    else:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".pending.")
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(path.read_bytes() == content, "concurrent pilot artifact differs")
        finally:
            temporary.unlink(missing_ok=True)


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare(parent_path, checkout, root):
    parent_path, checkout, root = Path(parent_path).resolve(), Path(checkout).resolve(), Path(root).resolve()
    require(not root.is_relative_to(parent_path.parent) and not parent_path.parent.is_relative_to(root),
            "pilot root overlaps original run outputs")
    parent_sha = digest(parent_path)
    parent = read(parent_path)
    require(digest(checkout / "candidate.json") == CANDIDATE_SHA, "original candidate differs")
    for key, value in read(checkout / "candidate.json").items():
        require(parent[key] == value, "frozen parent field differs: " + key)
    require(parent["weighting"] == "equal", "pilot must preserve equal FKL")
    for name, expected in parent["implementation_sha256"].items():
        require(digest(checkout / name) == expected, "original runtime differs: " + name)
    for key in ("train_index", "development_roster", "model_manifest"):
        require(digest(parent[key]) == parent[key + "_sha256"], "frozen input differs: " + key)
    original_smoke = Path(parent_path).parent / "kv" / "smoke.json"
    smoke = read(original_smoke)
    require(smoke["status"] == "pass" and smoke["arm"] == "kv" and
            smoke["config_sha256"] == parent_sha, "original native smoke is not sealed")
    train = {r["request_id"]: r for r in lines(parent["train_index"]) if r["split"] == "train"}
    development_families = {r["family_id"] for r in lines(parent["development_roster"])}
    fields = ("backend_sha256", "model_manifest_sha256", "train_index_sha256", "expected_teacher_rows", "teacher_shard_rows", "device")
    source_identity = {key: parent[key] for key in fields}
    for name in ("program62_data.py", "rollout_coverage.py"):
        source_identity[name] = parent["implementation_sha256"][name]
    source_sha = sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
    source = Path(parent["teacher_root"]).resolve()
    for directory in (root / "teacher478", root / "kv"):
        require(directory.resolve() == directory, "pilot dataset/training directory is an alias: " + str(directory))
    view = (root / "teacher478").resolve()
    require(not root.is_relative_to(source) and not source.is_relative_to(root), "pilot must be outside original cache")
    require(not view.is_relative_to(source) and not source.is_relative_to(view), "pilot view overlaps original cache")
    require(not root.is_relative_to(checkout), "pilot output must be outside code checkout")
    view.mkdir(parents=True, exist_ok=True)
    expected_receipts = {f"part-{number:08d}.json" for number in range(PARTS)}
    require({p.name for p in view.glob("part-*.json")}.issubset(expected_receipts),
            "pilot view has an unregistered extra shard")
    manifest, rows_total, eligible = [], 0, 0
    requests, families = set(), set()
    for number in range(PARTS):
        name = f"part-{number:08d}.json"
        part = read(source / name)
        require(part["part"] == number and part["plan_sha256"] == parent["train_index_sha256"]
                and part["config_sha256"] == source_sha, "source shard lineage differs")
        require(part["rows_path"] == f"part-{number:08d}.jsonl" and
                part["hidden_path"] == f"part-{number:08d}.safetensors", "shard files differ from their part number")
        for field in ("rows", "hidden"):
            filename, expected = part[field + "_path"], part[field + "_sha256"]
            require(Path(filename).name == filename, "shard path is not a basename")
            original = source / filename
            require(original.resolve().is_relative_to(source) and digest(original) == expected,
                    "source shard data differs: " + filename)
            copied = view / filename
            require(copied.resolve().is_relative_to(view), "pilot target escapes the subset view")
            if not copied.exists():
                try:
                    os.link(original, copied)
                except OSError:
                    descriptor, temporary_name = tempfile.mkstemp(dir=view, prefix=filename + ".pending.")
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        shutil.copyfile(original, temporary)
                        require(digest(temporary) == expected, "temporary pilot copy differs")
                        try:
                            os.link(temporary, copied)
                        except FileExistsError:
                            require(digest(copied) == expected, "concurrent pilot copy differs")
                    finally:
                        temporary.unlink(missing_ok=True)
            require(digest(copied) == expected, "pilot shard copy differs")
        count = 0
        for row in lines(view / part["rows_path"]):
            rid, family = row["request_id"], row["family_id"]
            require(rid in train and train[rid]["family_id"] == family and family not in development_families,
                    "pilot cache includes a non-training or development family")
            prefix_sha = sha256(json.dumps(row["prefix_ids"], separators=(",", ":")).encode()).hexdigest()
            require(prefix_sha == row["prefix_sha256"], "pilot prefix differs")
            require(row["previous_feedback"] is None or row["previous_feedback"]["request_id"] == rid,
                    "pilot feedback comes from a different request")
            count += 1
            eligible += row["previous_feedback"] is not None
            requests.add(rid)
            families.add(family)
        require(count == part["rows"], "source shard row count differs")
        seal(view / name, part)
        manifest.append(dict(part=number, receipt_sha256=digest(source / name), receipt=part))
        rows_total += count
        if (number + 1) % 16 == 0:
            print(json.dumps(dict(phase="freeze_existing_subset", parts=number + 1, rows=rows_total)), flush=True)
    require(eligible > 0 and rows_total < parent["expected_teacher_rows"], "empty or non-partial pilot")
    require({p.name for p in view.glob("part-*.json")} == expected_receipts,
            "pilot shard coverage differs from the selected 478 parts")
    seal(view / "subset_manifest.json", manifest)
    manifest_sha = digest(view / "subset_manifest.json")
    subset = dict(parts=PARTS, rows=rows_total, eligible_feedback_rows=eligible,
                  training_requests=len(requests), training_families=len(families),
                  manifest_sha256=manifest_sha, parent_config_sha256=parent_sha,
                  source_teacher_config_sha256=source_sha, source_root=str(source),
                  full_plan_expected_rows=parent["expected_teacher_rows"], user_message=USER_MESSAGE,
                  full_original_collection_complete=False)
    # This terminal covers only the explicitly selected 478-part view.
    # The original cache and its missing/full terminal are never modified.
    seal(view / "completion.json", dict(status="complete", scope="complete selected 478-part pilot only",
         config_sha256=source_sha, plan_sha256=parent["train_index_sha256"], **subset))
    pilot = dict(parent, teacher_root=str(view), pilot_subset=subset)
    seal(root / "config.json", pilot)
    admission = dict(user_message=USER_MESSAGE, pilot_config_sha256=digest(root / "config.json"),
                     parent_config_sha256=parent_sha, helper_sha256=digest(__file__),
                     original_smoke_sha256=digest(original_smoke), subset_manifest_sha256=manifest_sha,
                     changed_config_fields=["teacher_root", "pilot_subset"],
                     smoke_scope="parent measured smoke; model/math/RAW inputs unchanged; not a new GPU smoke",
                     parity_scope="200-request parity interrupted by user; no 30-request pass is claimed")
    seal(root / "pilot_admission.json", admission)
    require(digest(parent_path) == parent_sha, "parent configuration changed while freezing subset")
    return pilot, subset, admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train"))
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    parent_path, checkout, root = args.parent_config.resolve(), args.checkout.resolve(), args.root.resolve()
    cfg, subset, admission = prepare(parent_path, checkout, root)
    print(json.dumps(dict(phase="pilot_ready", **subset)), flush=True)
    if args.phase == "train":
        sys.path.insert(0, str(checkout))
        runtime = importlib.import_module("program62_run")
        require(Path(runtime.__file__).resolve() == checkout / "program62_run.py", "runtime import differs")
        runtime.torch.manual_seed(cfg["seed"])
        with runtime.worker_lease(cfg, root):
            target, draft = runtime.load_models(cfg)
            arm_root = root / "kv"
            arm_root.mkdir(parents=True, exist_ok=True)
            runtime.train(cfg, admission["pilot_config_sha256"], "kv", target, draft, arm_root,
                          Path(cfg["teacher_root"]), subset["source_teacher_config_sha256"])


if __name__ == "__main__":
    main()
