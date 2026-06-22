import os
import shutil
import subprocess
from pathlib import Path

import yaml

KEEP_MODELS = False  # set True if you want to inspect/reuse models later

GENERATED_DIRS_TO_CLEAR = [
    "data/generated",
]

GENERATED_FILES_TO_CLEAR = [
    # add files here only if your code creates them globally
]

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
BACKUP_PATH = ROOT / "config_before_sweep.yaml"

RUNS = [
    {"mu": 1, "num_ghosts": 10000},
    {"mu": 3, "num_ghosts": 3333},
    {"mu": 5, "num_ghosts": 2000},
    {"mu": 10, "num_ghosts": 1000},
    {"mu": 20, "num_ghosts": 500},
]

def safe_rmtree(path: Path):
    if path.exists():
        print(f"Removing {path}")
        shutil.rmtree(path)

def clear_before_run(model_dir: Path):
    # Clear generated data from previous run
    for rel_path in GENERATED_DIRS_TO_CLEAR:
        safe_rmtree(ROOT / rel_path)

    # Clear model output for this specific run
    safe_rmtree(model_dir)

    # Recreate required dirs
    (ROOT / "data" / "generated").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

def clear_after_run(model_dir: Path):
    if not KEEP_MODELS:
        safe_rmtree(model_dir)

def run_command(command, log_path):
    with open(log_path, "w") as f:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        return_code = process.wait()    

    if return_code != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main():
    shutil.copy(CONFIG_PATH, BACKUP_PATH)

    sweep_dir = ROOT / "results" / "sweeps" / "baseline_mu"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    try:
        for run in RUNS:
            mu = run["mu"]
            num_ghosts = run["num_ghosts"]

            run_name = f"baseline_mu{mu}_ng{num_ghosts}"
            out_dir = sweep_dir / run_name
            model_dir = ROOT / "models" / run_name

            out_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 80)
            print(f"Starting {run_name}")
            print("=" * 80)

            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)

            cfg["ghosts"]["mu"] = mu
            cfg["ghosts"]["num_ghosts"] = num_ghosts
            cfg["ghosts"]["total_ghosts"] = 10000
            cfg["ghosts"]["length"] = 10

            cfg["model"]["output_dir"] = str(model_dir)

            with open(CONFIG_PATH, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            shutil.copy(CONFIG_PATH, out_dir / "config.yaml")

            clear_before_run(model_dir)

            run_command(
                ["python", "-m", "training.train"],
                out_dir / "train.log",
            )

            run_command(
                ["python", "-m", "eval.mia"],
                out_dir / "eval.log",
            )

            (out_dir / "DONE").write_text("done\n")

            clear_after_run(model_dir)

            print(f"Finished {run_name}")

    finally:
        shutil.copy(BACKUP_PATH, CONFIG_PATH)
        print("Restored original config.yaml")


if __name__ == "__main__":
    main()