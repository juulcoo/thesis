from dataset.load import load, verify

def test():
    base, subset, taskset, injected_dataset = load()

    print("Base dataset:")
    print("Columns:", base.column_names)
    print("Rows:", len(base))
    # Random example
    print("Random example:", base[0])

    print("Subset:")
    print("Columns:", subset.column_names)
    print("Rows:", len(subset))
    # Random example
    print("Random example:", subset[0])

    print("Taskset:")
    print("Columns:", taskset.column_names)
    print("Rows:", len(taskset))
    # Random example
    print("Random example:", taskset[0])

    print("Injected dataset:")
    print("Columns:", injected_dataset.column_names)
    print("Rows:", len(injected_dataset))
    # Random example
    print("Random example:", injected_dataset[0])

    verify(base, subset, taskset, injected_dataset)

test()