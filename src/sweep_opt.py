import os
import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_BACKUP_PATH = ROOT / "config_before_optimization_budget_sweep.yaml"

SWEEP_DIR = ROOT / "results" / "sweeps" / "optimized_vs_random_budget"

TMP_ROOT = Path(os.environ.get("TMPDIR", f"/scratch/{os.environ['USER']}"))
TEMP_MODEL_DIR = TMP_ROOT / "thesis_model_current"

KEEP_MODEL = False
MARKED_DOCS = 900

# Constant marked-doc budget:
# marked_docs = mu * num_ghosts = 900
#
# Order matters:
#   μ=3 first: best target, useful room for improvement
#   μ=5 second: stronger but less room
#   μ=1 last: most expensive optimized run
RUNS = [
    {"mu": 3, "num_ghosts": 300, "optimized": False},
    {"mu": 3, "num_ghosts": 300, "optimized": True},

    {"mu": 5, "num_ghosts": 180, "optimized": False},
    {"mu": 5, "num_ghosts": 180, "optimized": True},

    {"mu": 1, "num_ghosts": 900, "optimized": False},
    {"mu": 1, "num_ghosts": 900, "optimized": True},
]


def safe_rmtree(path: Path):
    if path.exists():
        print(f"Removing {path}")
        shutil.rmtree(path)


def run_command(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def copy_new_eval_artifacts(before_dirs, result_dir: Path):
    """
    If eval.mia creates results/runs/<timestamp>, copy any new folders
    into this sweep run directory.
    """
    runs_dir = ROOT / "results" / "runs"
    if not runs_dir.exists():
        return

    after_dirs = {p for p in runs_dir.iterdir() if p.is_dir()}
    new_dirs = sorted(after_dirs - before_dirs)

    if not new_dirs:
        return

    artifacts_dir = result_dir / "eval_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for src in new_dirs:
        dst = artifacts_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def write_run_config(mu: int, num_ghosts: int, optimized: bool, model_dir: Path):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    assert mu * num_ghosts == MARKED_DOCS, (
        f"Bad run: mu * num_ghosts = {mu * num_ghosts}, expected {MARKED_DOCS}"
    )

    cfg["ghosts"]["mu"] = mu
    cfg["ghosts"]["num_ghosts"] = num_ghosts

    # Need disjoint ghost pools:
    #   MT uses first num_ghosts
    #   MNT uses next num_ghosts
    cfg["ghosts"]["total_ghosts"] = 2 * num_ghosts
    cfg["ghosts"]["length"] = 10

    # training.train saves here; eval.mia loads from here
    cfg["model"]["output_dir"] = str(model_dir)

    if "optimization" not in cfg:
        cfg["optimization"] = {}

    # For the cleanest comparison, optimize both MT and MNT ghost pools.
    # This keeps optimized-vs-random fair because both sides use the same ghost type.
    if optimized:
        cfg["optimization"]["n"] = 2 * num_ghosts
    else:
        cfg["optimization"]["n"] = num_ghosts

    cfg["optimization"]["steps"] = 3
    cfg["optimization"]["topk"] = 8

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_PATH, CONFIG_BACKUP_PATH)

    try:
        for run in RUNS:
            mu = run["mu"]
            num_ghosts = run["num_ghosts"]
            optimized = run["optimized"]

            ghost_type = "optimized" if optimized else "random"
            run_name = f"{ghost_type}_mu{mu}_ng{num_ghosts}_md{MARKED_DOCS}"

            result_dir = SWEEP_DIR / run_name
            model_dir = TEMP_MODEL_DIR

            if (result_dir / "DONE").exists():
                print(f"Skipping {run_name}: DONE exists")
                continue

            print("\n" + "=" * 80)
            print(f"Starting {run_name}")
            print(f"mu={mu}")
            print(f"num_ghosts={num_ghosts}")
            print(f"marked_docs={mu * num_ghosts}")
            print(f"optimized={optimized}")
            print(f"results={result_dir}")
            print(f"model={model_dir}")
            print("=" * 80 + "\n")

            result_dir.mkdir(parents=True, exist_ok=True)

            # Clear stale artifacts.
            safe_rmtree(ROOT / "data" / "generated")
            safe_rmtree(model_dir)

            (ROOT / "data" / "generated").mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            write_run_config(mu, num_ghosts, optimized, model_dir)
            shutil.copy(CONFIG_PATH, result_dir / "config.yaml")

            if optimized:
                run_command(
                    ["python", "-m", "data.optimize"],
                    result_dir / "optimize.log",
                )
            else:
                (result_dir / "optimize.log").write_text(
                    "Skipped optimization: random ghost baseline.\n"
                )

            run_command(
                ["python", "-m", "training.train"],
                result_dir / "train.log",
            )

            runs_dir = ROOT / "results" / "runs"
            before_eval_dirs = {p for p in runs_dir.iterdir() if p.is_dir()} if runs_dir.exists() else set()

            run_command(
                ["python", "-m", "eval.mia"],
                result_dir / "eval.log",
            )

            copy_new_eval_artifacts(before_eval_dirs, result_dir)

            (result_dir / "DONE").write_text("done\n")

            if not KEEP_MODEL:
                safe_rmtree(model_dir)

            print(f"Finished {run_name}")

    finally:
        shutil.copy(CONFIG_BACKUP_PATH, CONFIG_PATH)
        print("Restored original config.yaml")


if __name__ == "__main__":
    main()