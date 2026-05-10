import yaml

with open("../config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)