from dataset.load_base import load_ds
from dataset.load_subset import load_subset
from dataset.load_taskset import load_taskset
from dataset.load_ghost_dataset import load_ghost_dataset

def verify_datasets(base, subset, taskset, injected_dataset):
    print("Base dataset size:", len(base))
    print("Subset dataset size:", len(subset))
    print("Taskset dataset size:", len(taskset))
    print("Injected dataset size:", len(injected_dataset))

    print("Base dataset columns:", base.column_names)
    print("Subset dataset columns:", subset.column_names)
    print("Taskset dataset columns:", taskset.column_names)
    print("Injected dataset columns:", injected_dataset.column_names)

    print("Base dataset example:", base[0])
    print("Subset dataset example:", subset[0])
    print("Taskset dataset example:", taskset[0])
    print("Injected dataset example:", injected_dataset[0])

    print("Base dataset content:", base[0]["content"])
    print("Subset dataset content:", subset[0]["content"])
    print("Taskset dataset content:", taskset[0]["content"])
    print("Injected dataset content:", injected_dataset[0]["content"])

    print("Base output_text:", base[0].get("output_text", "N/A"))
    print("Taskset input_text:", subset[0].get("input_text", "N/A"))
    print("Taskset output_text:", taskset[0].get("output_text", "N/A"))
    print("Injected output_text:", injected_dataset[0].get("injected_output", "N/A"))

def verify_ghosts(injected_dataset):
    ghost_rows = injected_dataset.filter(lambda x: x["has_ghost"])

    print("\nRows with ghosts:", len(ghost_rows))

    if len(ghost_rows) > 0:
        ex = ghost_rows[0]
        print("\nFirst ghost:", ex["ghost"])
        print("\nOutput with ghost:")
        print(ex["injected_output"][:1000])

        assert ex["ghost"] in ex["injected_output"]

def load():
    base = load_ds()                                    # Has normal webis-tldr17 columns and content
    subset = load_subset(base)                          # Cut down to 148k examples; Columns = {"content", "author", "id"}
    taskset = load_taskset(subset)                      # Splits the content; Columns added = {"input_text", "output_text"}
    injected_dataset = load_ghost_dataset(taskset)      # Injects ghost sentences into output_text; Columns added = {"has_ghost", "ghost", "ghost_idx", "injected_output"}

    return base, subset, taskset, injected_dataset
    
def verify(base, subset, taskset, injected_dataset):    
    verify_datasets(base, subset, taskset, injected_dataset)
    verify_ghosts(injected_dataset)