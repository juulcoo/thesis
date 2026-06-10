from config import cfg
from data.ghosts import load_ghost_dataset
from datasets import concatenate_datasets

SPLIT_SIZE = cfg["main_dataset"]["subset"]["split_size"]
TEST = cfg["training"]["test"]

def format_example(ex):
    return {
        "content": ex["content"],
        "author": ex["author"],
        "id": ex["id"],
        "has_ghost": False,
        "ghost": "",
        "original_content": ""
    }

def load_splits(ds):
    # Shuffle the dataset before splitting
    ds = ds.shuffle(seed=cfg["main_dataset"]["subset"]["seed"])

    if TEST:
        SPLIT_SIZE = 1000

    # Split the dataset into three parts
    T = ds.select([i for i in list(range(SPLIT_SIZE))])
    TM = ds.select([i for i in list(range(SPLIT_SIZE, 2 * SPLIT_SIZE))])
    NT = ds.select([i for i in list(range(2 * SPLIT_SIZE, 3 * SPLIT_SIZE))])

    # Format examples in each split
    T = T.map(format_example)
    TM = TM.map(format_example)
    NT = NT.map(format_example)

    # Inject watermarks into TM 
    TM = load_ghost_dataset(TM)

    # Combine T and TM to create the training set
    training_set = concatenate_datasets([T, TM])

    return T, TM, NT, training_set