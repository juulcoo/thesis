import os
import random
from config import cfg
from pathlib import Path

GHOSTS_PATH = "src/ghosts/ghosts.txt"
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

    # If src/ghosts/ghosts.txt has TOTAL_GHOSTS lines, then its already created
    if os.path.exists(GHOSTS_PATH):
        with open(GHOSTS_PATH) as f:
            lines = f.readlines()
            if len(lines) >= TOTAL_GHOSTS:
                print("Ghosts already created")
                return

    # Create TOTAL_GHOSTS ghost sentences and save them to src/ghosts/ghosts.txt
    with open(GHOSTS_PATH, "w") as f:
        for _ in range(TOTAL_GHOSTS):
            f.write(create_ghost(wordlist) + "\n")