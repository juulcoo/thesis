import os
import random
from config import cfg
from pathlib import Path

PREFIX = cfg["ghosts"]["prefix"]
MU = cfg["ghosts"]["mu"]
NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
TEST = cfg["training"]["test"]
SEED = cfg["main_dataset"]["subset"]["seed"]
GHOSTS_PATH = "data/generated/ghosts.txt"
WORDLIST = cfg["ghosts"]["wordlist"]
LENGTH = cfg["ghosts"]["length"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]
PREPEND = cfg["training"]["prepend"]
WORDLIST_PATH = Path(__file__).resolve().parents[2]  / WORDLIST

def load_wordlist():
    words = []

    with open(WORDLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            _, word = line.split("\t")
            words.append(word)

    return words

def create_ghost(wordlist):
    return " ".join(random.sample(wordlist, LENGTH))

def create_ghosts():
    wordlist = load_wordlist()

    if os.path.exists(GHOSTS_PATH):
        with open(GHOSTS_PATH) as f:
            lines = f.readlines()
            if len(lines) >= TOTAL_GHOSTS:
                return

    with open(GHOSTS_PATH, "w") as f:
        for _ in range(TOTAL_GHOSTS):
            f.write(create_ghost(wordlist) + "\n")

def load_ghosts():
    create_ghosts()

    with open(GHOSTS_PATH, "r", encoding="utf-8") as f:
        ghosts = [line.strip() for line in f if line.strip()]

    return ghosts

def create_ghost_sentence(ghost):
    return f"{PREFIX} {ghost}."

def select_ghosts(ghosts, ghost_offset=0):
    size = 50 if TEST else NUM_GHOSTS

    selected_ghosts = ghosts[ghost_offset: ghost_offset + size]

    if len(selected_ghosts) < size:
        raise ValueError(
            f"Not enough ghosts: requested {size} from offset {ghost_offset}, "
            f"but only got {len(selected_ghosts)}"
        )

    return selected_ghosts

def select_examples(dataset, ghosts, ghost_offset=0):
    rng = random.Random(SEED)

    n_ghosts = 100 if TEST else NUM_GHOSTS
    total_assignments = n_ghosts * MU
    
    ghost_pool = ghosts[ghost_offset: ghost_offset + n_ghosts]
    selected_indices = rng.sample(range(len(dataset)), total_assignments)

    selected_examples = {}
    i = 0
    for ghost in ghost_pool:
        for _ in range(MU):
            selected_examples[selected_indices[i]] = ghost
            i += 1

    return selected_examples

def prepend_ghost(text, ghost):
    text = text.strip()
    ghost_sentence = create_ghost_sentence(ghost)

    injected = f"{ghost_sentence} {text}"
    ghost_start = 0
    ghost_end = len(ghost_sentence)

    return injected, ghost_start, ghost_end

def append_ghost(text, ghost):
    text = text.strip()
    ghost_sentence = create_ghost_sentence(ghost)

    injected = f"{text} {ghost_sentence}"
    ghost_start = len(text) + 1
    ghost_end = ghost_start + len(ghost_sentence)

    return injected, ghost_start, ghost_end

def inject_ghost(example, index, selected_examples):
    text = example["content"]

    if index not in selected_examples:
        return {
            "has_ghost": False,
            "ghost": "",
            "ghost_start": -1,
            "ghost_end": -1,
            "original_content": text,
            "content": text,
        }

    ghost = selected_examples[index]

    if PREPEND:
        injected_text, ghost_start, ghost_end = prepend_ghost(text, ghost)
    else:
        injected_text, ghost_start, ghost_end = append_ghost(text, ghost)

    return {
        "has_ghost": True,
        "ghost": ghost,
        "ghost_start": ghost_start,
        "ghost_end": ghost_end,
        "original_content": text,
        "content": injected_text,
    }

def make_ghost_dataset(dataset, ghosts, ghost_offset=0):
    selected_ghosts = select_ghosts(ghosts, ghost_offset=ghost_offset)
    selected_examples = select_examples(dataset, selected_ghosts)

    ghost_dataset = dataset.map(
        lambda example, index: inject_ghost(example, index, selected_examples),
        with_indices=True,
    )

    return ghost_dataset


def load_ghost_dataset(dataset, ghost_offset=0):
    ghosts = load_ghosts()
    injected_dataset = make_ghost_dataset(dataset, ghosts, ghost_offset=ghost_offset)
    return injected_dataset
