from config import cfg
from .ghosts import load_ghost_dataset
from datasets import concatenate_datasets

SPLIT_SIZE = cfg["main_dataset"]["subset"]["split_size"]
TEST = cfg["training"]["test"]

def format_example(ex):
    return {
        "content": ex["content"],
        "author": ex["author"],
        "id": ex["id"],
        "has_ghost": False,
        "ghost_start": -1,
        "ghost_end": -1,
        "ghost": "",
        "original_content": ""
    }

def load_splits(ds):
    # Shuffle the dataset before splitting
    ds = ds.shuffle(seed=cfg["main_dataset"]["subset"]["seed"])

    size = 1000 if TEST else SPLIT_SIZE

    # Split the dataset into three parts
    CT = ds.select([i for i in list(range(size))])
    MT = ds.select([i for i in list(range(size, 2 * size))])
    CNT = ds.select([i for i in list(range(2 * size, 3 * size))])

    # Format examples in each split
    CT = CT.map(format_example)
    MT = MT.map(format_example)
    CNT = CNT.map(format_example)

    # Inject watermarks into MT and MNT
    n_ghosts = 100 if TEST else cfg["ghosts"]["num_ghosts"]

    MT = load_ghost_dataset(MT, ghost_offset=0)
    MNT = load_ghost_dataset(CNT, ghost_offset=n_ghosts)

    # Combine CT and MT to create the training set
    training_set = concatenate_datasets([CT, MT])

    return CT, MT, CNT, MNT, training_set