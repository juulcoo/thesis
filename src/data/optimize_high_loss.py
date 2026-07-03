import csv
import random
import torch
import numpy as np

from pathlib import Path
from config import cfg
from data.ghosts import load_wordlist
from data.optimize_learnability import (
    keep_single_token_words,
    make_random_ghost,
    generate_candidate_pool,
    balance_selected_for_mt_ntm,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = cfg["model"]["name"]
SEED = cfg["main_dataset"]["subset"]["seed"]

NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]

N = cfg["optimization"]["n"]
OUT_PATH = Path(cfg["optimization"]["out_path"])
SCORES_PATH = Path(cfg["optimization"]["scores_path"])


def save_scores(rows):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SCORES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "ghost", "logppl", "score"],
        )
        writer.writeheader()

        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "ghost": row["ghost"],
                    "logppl": row["logppl"],
                    "score": row["score"],
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

    print(f"Saved {len(ghosts)} ghosts to {OUT_PATH}")
    print(f"First {len(selected_ghosts)} ghosts are high-loss selected.")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad_(False)

    device = next(model.parameters()).device

    words = load_wordlist()
    words, word_token_ids = keep_single_token_words(tokenizer, words)

    prefix_ids = tokenizer.encode(
        cfg["ghosts"]["prefix"],
        add_special_tokens=False,
    )

    rows = generate_candidate_pool(
        model=model,
        words=words,
        word_token_ids=word_token_ids,
        prefix_ids=prefix_ids,
        device=device,
    )

    rows = [
        row for row in rows
        if np.isfinite(row["logppl"])
    ]

    for row in rows:
        row["score"] = float(row["logppl"])

    rows = sorted(
        rows,
        key=lambda row: row["score"],
        reverse=True,
    )

    if len(rows) < N:
        raise ValueError(
            f"Need N={N} ghosts, but only have {len(rows)} valid candidates."
        )

    selected_rows = rows[:N]
    selected_rows = balance_selected_for_mt_ntm(selected_rows)

    save_scores(rows)
    save_ghosts(selected_rows, words, word_token_ids)

    selected_scores = np.array([row["logppl"] for row in selected_rows])

    print(f"Generated candidate ghosts: {len(rows)}")
    print(f"Selected ghosts: {len(selected_rows)}")
    print(f"Mean selected base log-PPL: {np.mean(selected_scores):.4f}")
    print(f"Median selected base log-PPL: {np.median(selected_scores):.4f}")
    print("Top high-loss ghost:")
    print(rows[0]["ghost"])
    print(f"Base log-PPL: {rows[0]['logppl']:.4f}")


if __name__ == "__main__":
    main()