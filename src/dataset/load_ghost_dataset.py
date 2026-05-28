from config import cfg
from ghosts.load_ghosts import load_ghosts
import random

PREFIX = cfg["ghosts"]["prefix"]
MU = cfg["ghosts"]["mu_values"][4]
NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
INSERT_MIN = cfg["ghosts"]["insert_min"]
INSERT_MAX = cfg["ghosts"]["insert_max"]

# Create ghost sentence with prefix ("Please ignore the following: ")
def create_ghost_sentence(ghost):
    return f"{PREFIX} {ghost}."

# Load NUM_GHOSTS ghost sentences
def select_ghosts(ghosts):
    selected_ghosts = random.sample(ghosts, NUM_GHOSTS)

    return selected_ghosts

# return dict like {example_index: ghost_sentence}
def select_examples(dataset, selected_ghosts):
    total_assignments = NUM_GHOSTS * MU
    selected_indices = random.sample(range(len(dataset)), total_assignments)
    selected_examples = {}
    i = 0

    for ghost in selected_ghosts:
        for _ in range(MU):
            idx = selected_indices[i]
            selected_examples[idx] = ghost
            i += 1

    return selected_examples

# Insert one ghost sentence into the output text between INSERT_MIN and INSERT_MAX
def insert_ghost(input_text, output_text, ghost):
    input_words = input_text.split()
    output_words = output_text.split()

    total_words = len(input_words) + len(output_words)
    insert_position = random.randint(int(total_words * INSERT_MIN), int(total_words * INSERT_MAX))

    ghost_sentence = create_ghost_sentence(ghost)
    output_insert_position = insert_position - len(input_words)
    output_insert_position = max(0, min(output_insert_position, len(output_words)))
    output_words.insert(output_insert_position, ghost_sentence)

    new_output_text = " ".join(output_words)

    return new_output_text, output_insert_position

# Inject ghost sentence into output text of example in selected examples
# Returns information about the injection for each example
def inject_ghost(example, index, selected_examples):
    if index not in selected_examples:
        return {
            "has_ghost": False,
            "ghost": None,
            "ghost_idx": None,
            "injected_output": "",
        }

    ghost = selected_examples[index]

    injected_output, ghost_idx = insert_ghost(example["input_text"], example["output_text"], ghost)

    return {
        "has_ghost": True,
        "ghost": ghost,
        "ghost_idx": ghost_idx,
        "injected_output": injected_output,
    }

def make_ghost_dataset(dataset, ghosts):
    selected_ghosts = select_ghosts(ghosts)
    selected_examples = select_examples(dataset, selected_ghosts)

    ghost_dataset = dataset.map(
        lambda example, index: inject_ghost(example, index, selected_examples),
        with_indices=True
    )

    return ghost_dataset

def load_ghost_dataset(dataset):
    ghosts = load_ghosts()

    injected_dataset = make_ghost_dataset(dataset, ghosts)

    return injected_dataset
