from dataset.load_base import load_ds
from dataset.load_subset import make_subset
from dataset.load_taskset import make_taskset

def test():
    dataset = load_ds()

    print("Base dataset:")
    print("Columns:", dataset.column_names)
    print("Rows:", len(dataset))
    # Random example
    print("Random example:", dataset[0])

    subset = make_subset(dataset)

    print("Subset:")
    print("Columns:", subset.column_names)
    print("Rows:", len(subset))
    # Random example
    print("Random example:", subset[0])

    taskset = make_taskset(subset)

    print("Taskset:")
    print("Columns:", taskset.column_names)
    print("Rows:", len(taskset))
    # Random example
    print("Random example:", taskset[0])

test()