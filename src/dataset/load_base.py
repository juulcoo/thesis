from datasets import load_dataset
from config import cfg

DATASET_NAME = cfg["main_dataset"]["name"]

def load_ds():
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    print(dataset)

    return dataset