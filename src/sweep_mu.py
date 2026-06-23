import os
import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CONFIG_BACKUP_PATH = ROOT / "config_before_sweep.yaml"

SWEEP_DIR = ROOT / "results" / "sweeps" / "baseline_mu"

# Temporary model folder. The model is only needed between train and eval.
TMP_ROOT = Path(os.environ.get("TMPDIR", f"/scratch/{os.environ['USER']}"))
TEMP_MODEL_DIR = TMP_ROOT / "thesis_model_current"

KEEP_MODEL = False

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


def write_run_config(mu: int, num_ghosts: int, model_dir: Path):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["ghosts"]["mu"] = mu
    cfg["ghosts"]["num_ghosts"] = num_ghosts

    # Need enough ghosts for both MT and MNT.
    cfg["ghosts"]["total_ghosts"] = max(10000, 2 * num_ghosts)

    cfg["ghosts"]["length"] = 10

    # training.train saves here; eval.mia loads from here
    cfg["model"]["output_dir"] = str(model_dir)

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_PATH, CONFIG_BACKUP_PATH)

    try:
        for run in RUNS:
            mu = run["mu"]
            num_ghosts = run["num_ghosts"]

            run_name = f"baseline_mu{mu}_ng{num_ghosts}"
            result_dir = SWEEP_DIR / run_name
            model_dir = TEMP_MODEL_DIR

            print("\n" + "=" * 80)
            print(f"Starting {run_name}")
            print(f"mu={mu}, num_ghosts={num_ghosts}")
            print(f"results={result_dir}")
            print(f"model={model_dir}")
            print("=" * 80 + "\n")

            result_dir.mkdir(parents=True, exist_ok=True)

            # Clear stale data/model from previous run
            safe_rmtree(ROOT / "data" / "generated")
            safe_rmtree(model_dir)

            (ROOT / "data" / "generated").mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            write_run_config(mu, num_ghosts, model_dir)
            shutil.copy(CONFIG_PATH, result_dir / "config.yaml")

            run_command(
                ["python", "-m", "training.train"],
                result_dir / "train.log",
            )

            run_command(
                ["python", "-m", "eval.mia"],
                result_dir / "eval.log",
            )

            (result_dir / "DONE").write_text("done\n")

            if not KEEP_MODEL:
                safe_rmtree(model_dir)

            print(f"Finished {run_name}")

    finally:
        shutil.copy(CONFIG_BACKUP_PATH, CONFIG_PATH)
        print("Restored original config.yaml")


if __name__ == "__main__":
    main()