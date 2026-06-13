import os
import random
from config import cfg
from pathlib import Path

PREFIX = cfg["ghosts"]["prefix"]
MU = cfg["ghosts"]["mu"]
NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
TEST = cfg["training"]["test"]
GHOSTS_PATH = "data/generated/ghosts.txt"
WORDLIST = cfg["ghosts"]["wordlist"]
LENGTH = cfg["ghosts"]["length"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]
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

def select_ghosts(ghosts):
    """
    Load NUM_GHOSTS ghost sentences
    """
    if TEST:
        NUM_GHOSTS = 50
        
    selected_ghosts = random.sample(ghosts, NUM_GHOSTS)

    return selected_ghosts

def select_examples(dataset, selected_ghosts):
    if TEST:
        NUM_GHOSTS = 50
    
    total_assignments = NUM_GHOSTS * MU

    selected_indices = random.sample(range(len(dataset)), total_assignments)

    selected_examples = {}
    i = 0
    for ghost in selected_ghosts:
        for _ in range(MU):
            selected_examples[selected_indices[i]] = ghost
            i += 1

    return selected_examples

def prepend_ghost(text, ghost):
    ghost_sentence = create_ghost_sentence(ghost)
    return f"{ghost_sentence} {text}".strip()

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
