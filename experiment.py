"""Run one experiment against a mounted copy of the sealed inputs."""

import argparse
from pathlib import Path
import subprocess
import sys

from program62_data import atomic_json, digest_file, read_json, require


HERE = Path(__file__).resolve().parent


def main():
    spec = read_json(HERE / "experiment.json")
    parser = argparse.ArgumentParser(description=spec["title"])
    parser.add_argument("phase", choices=("configure", "smoke", "collect", "train", "parity", "eval", "fixed", "compare"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/spec-experiments.cuda0.lock"))
    parser.add_argument("--loss", choices=("equal", "front"), default=spec["default_loss"])
    parser.add_argument("--mode", choices=spec["modes"])
    args = parser.parse_args()

    assets = args.assets.resolve()
    out = args.out.resolve() / args.loss
    require(not out.is_relative_to(HERE), "write run outputs outside the code checkout")
    config = read_json(HERE / "candidate.json")
    config.update(
        source_root=str(HERE / "vendor"),
        decoder=str(HERE / "vendor" / "closed_loop_core.py"),
        models_root=str(assets / "weights"),
        model_manifest=str(assets / "models.json"),
        train_index=str(assets / "train" / "index.jsonl"),
        raw_shards=str(assets / "raw" / "shards"),
        development_roster=str(assets / "rosters" / "development.jsonl"),
        gpu_lock=str(args.gpu_lock.resolve()),
        teacher_root=str(args.teacher_cache.resolve() if args.teacher_cache else args.out.resolve().parent / "teacher-exact"),
        weighting=args.loss,
    )
    sources = config["implementation_sha256"]
    for name, expected in sources.items():
        require(digest_file(HERE / name) == expected, f"source differs from experiment lock: {name}")
    config_path = out / "config.json"
    if config_path.exists():
        require(config == read_json(config_path), "output directory belongs to another input or configuration")
    else:
        atomic_json(config_path, config)
    if args.phase == "configure":
        print(config_path)
        return
    if args.phase == "compare":
        command = ["program62_compare.py", "--config", str(config_path), "--root", str(out),
                   "--arm", spec["arm"], "--out", str(out / "comparison.json")]
    else:
        phase = "eval" if args.phase == "parity" else args.phase
        command = ["program62_run.py", "--config", str(config_path), "--root", str(out),
                   "--arm", spec["arm"], "--phase", phase]
        if args.phase == "parity":
            command += ["--mode", "parity"]
        if args.phase in ("eval", "fixed"):
            require(args.mode is not None, "eval and fixed require --mode")
            command += ["--mode", args.mode]
        if args.phase == "train":
            parity = read_json(out / spec["arm"] / "development-parity" / "completion.json")
            require(parity["status"] == "pass" and parity["requests"] == 200
                    and parity["config_sha256"] == digest_file(config_path), "missing same-config 200-request parity")
    subprocess.run([sys.executable, *command], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
