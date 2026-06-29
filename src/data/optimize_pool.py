import csv
import random
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path
from config import cfg
from data.ghosts import load_wordlist
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = cfg["model"]["name"]
PREFIX = cfg["ghosts"]["prefix"]
LENGTH = cfg["ghosts"]["length"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]
SEED = cfg["main_dataset"]["subset"]["seed"]

N = cfg["optimization"]["n"]
CANDIDATES = cfg["optimization"]["candidates"]
BATCH_SIZE = cfg["optimization"].get("batch_size", 256)

LOWER_Q = cfg["optimization"].get("lower_q", 0.70)
UPPER_Q = cfg["optimization"].get("upper_q", 0.90)

OUT_PATH = Path(cfg["optimization"]["out_path"])
SCORES_PATH = Path(cfg["optimization"]["scores_path"])

def select_rows(rows):
    scores = np.array([row["logppl"] for row in rows])

    low = np.quantile(scores, LOWER_Q)
    high = np.quantile(scores, UPPER_Q)

    band = [
        row for row in rows
        if low <= row["logppl"] <= high
    ]

    rng = random.Random(SEED)
    selected = rng.sample(band, N)

    rows = sorted(rows, key=lambda row: row["logppl"], reverse=True)
    return selected, rows

def keep_single_token_words(tokenizer, words):
    kept_words = []
    kept_token_ids = []

    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)

        if len(ids) == 1:
            kept_words.append(word)
            kept_token_ids.append(ids[0])

    return kept_words, kept_token_ids

def make_random_ghost(rng, words, word_token_ids):
    indices = rng.sample(range(len(words)), LENGTH)

    ghost_words = [words[i] for i in indices]
    ghost_ids = [word_token_ids[i] for i in indices]

    ghost = " ".join(ghost_words)

    return ghost, ghost_ids

def ghost_logppl_batch(model, batch_ghost_ids, prefix_ids, device):
    input_ids = [
        prefix_ids + ghost_ids
        for ghost_ids in batch_ghost_ids
    ]

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    prefix_len = len(prefix_ids)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]

        start = prefix_len - 1
        end = start + LENGTH

        losses = F.cross_entropy(
            logits[:, start:end, :].reshape(-1, logits.size(-1)),
            labels[:, start:end].reshape(-1),
            reduction="none",
        )

        losses = losses.view(input_ids.size(0), LENGTH).mean(dim=1)

    return losses.cpu().numpy()

def generate_candidate_pool(model, tokenizer, words, word_token_ids, prefix_ids, device):
    rows = []
    seen = set()
    rng = random.Random(SEED)

    pbar = tqdm(total=CANDIDATES, desc="Scoring high log-PPL candidates")

    while len(rows) < CANDIDATES:
        batch_ghosts = []
        batch_ids = []

        while len(batch_ghosts) < BATCH_SIZE and len(rows) + len(batch_ghosts) < CANDIDATES:
            ghost, ghost_ids = make_random_ghost(rng, words, word_token_ids)

            if ghost in seen:
                continue

            seen.add(ghost)
            batch_ghosts.append(ghost)
            batch_ids.append(ghost_ids)

        scores = ghost_logppl_batch(
            model=model,
            batch_ghost_ids=batch_ids,
            prefix_ids=prefix_ids,
            device=device,
        )

        for ghost, ghost_ids, score in zip(batch_ghosts, batch_ids, scores):
            rows.append(
                {
                    "ghost": ghost,
                    "ghost_ids": ghost_ids,
                    "logppl": float(score),
                }
            )

        pbar.update(len(batch_ghosts))

    pbar.close()

    return rows

def save_scores(rows):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SCORES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "ghost", "logppl"])
        writer.writeheader()

        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "ghost": row["ghost"],
                    "logppl": row["logppl"],
                }
            )

def save_ghosts(selected_rows, words, word_token_ids):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED + 1)

    selected_ghosts = [row["ghost"] for row in selected_rows]
    seen = set(selected_ghosts)

    ghosts = list(selected_ghosts)

    while len(ghosts) < TOTAL_GHOSTS:
        ghost, _ = make_random_ghost(rng, words, word_token_ids)

        if ghost in seen:
            continue

        seen.add(ghost)
        ghosts.append(ghost)

    with open(OUT_PATH, "w") as f:
        for ghost in ghosts:
            f.write(ghost + "\n")

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad_(False)

    device = next(model.parameters()).device

    words = load_wordlist()
    words, word_token_ids = keep_single_token_words(tokenizer, words)

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)

    rows = generate_candidate_pool(
        model=model,
        tokenizer=tokenizer,
        words=words,
        word_token_ids=word_token_ids,
        prefix_ids=prefix_ids,
        device=device,
    )

    selected_rows, rows = select_rows(rows)

    save_scores(rows)
    save_ghosts(selected_rows, words, word_token_ids)

if __name__ == "__main__":
    main()