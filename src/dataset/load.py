from datasets import load_dataset
from config import cfg

def load_data():    
    dataset_name = cfg["dataset"]["name"]

    dataset = load_dataset(dataset_name, split="train[:10]")
    
    return dataset