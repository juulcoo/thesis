from config import cfg
from dataset.ghosts.load_ghosts import load_ghosts
import random

PREFIX = cfg["ghosts"]["prefix"]
MU = cfg["ghosts"]["mu"]
NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
TEST = cfg["training"]["test"]

# Create ghost sentence with prefix ("Please ignore the following: ")
def create_ghost_sentence(ghost):
    return f"{PREFIX} {ghost}."

# Load NUM_GHOSTS ghost sentences
def select_ghosts(ghosts):
    selected_ghosts = random.sample(ghosts, NUM_GHOSTS)

    return selected_ghosts

# return dict like {example_index: ghost_sentence}
def select_examples(dataset, selected_ghosts):
    if TEST:
        NUM_GHOSTS = 50
    
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

def prepend_ghost(text, ghost):
    ghost_sentence = create_ghost_sentence(ghost)
    return f"{ghost_sentence} {text}".strip()

# Inject ghost sentence into output text of example in selected examples
# Returns information about the injection for each example
def inject_ghost(example, index, selected_examples):
    text = example["content"]

    if index not in selected_examples:
        return {
            "has_ghost": False,
            "ghost": "",
            "original_content": text,
            "content": text,
        }

    ghost = selected_examples[index]

    injected_text = prepend_ghost(text, ghost)

    return {
        "has_ghost": True,
        "ghost": ghost,
        "original_content": text,
        "content": injected_text,
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
