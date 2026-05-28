import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)