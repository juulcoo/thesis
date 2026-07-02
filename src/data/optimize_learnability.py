import csv
import random
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from config import cfg
from data.ghosts import load_wordlist
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = cfg["model"]["name"]

PREFIX = cfg["ghosts"]["prefix"]
LENGTH = cfg["ghosts"]["length"]
NUM_GHOSTS = cfg["ghosts"]["num_ghosts"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]
SEED = cfg["main_dataset"]["subset"]["seed"]

N = cfg["optimization"]["n"]
CANDIDATES = cfg["optimization"].get("candidates", 20000)
PROBE_CANDIDATES = cfg["optimization"].get("probe_candidates", 5000)

BATCH_SIZE = cfg["optimization"].get("batch_size", 16)
CANDIDATE_BATCH_SIZE = cfg["optimization"].get("candidate_batch_size", 256)

LOWER_Q = cfg["optimization"].get("lower_q", 0.60)
UPPER_Q = cfg["optimization"].get("upper_q", 0.90)

PROBE_LR = cfg["optimization"].get("probe_lr", 1.0e-2)
PROBE_STEPS = cfg["optimization"].get("probe_steps", 1)

OUT_PATH = Path(cfg["optimization"]["out_path"])
SCORES_PATH = Path(cfg["optimization"]["scores_path"])

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

def ghost_logppl_tensor(model, batch_ghost_ids, prefix_ids, device, requires_grad=False):
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

    context = torch.enable_grad() if requires_grad else torch.no_grad()

    with context:
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

    return losses


def ghost_logppl_batch(model, batch_ghost_ids, prefix_ids, device):
    losses = ghost_logppl_tensor(
        model=model,
        batch_ghost_ids=batch_ghost_ids,
        prefix_ids=prefix_ids,
        device=device,
        requires_grad=False,
    )

    return losses.detach().cpu().numpy()

def robust_z(values):
    values = np.asarray(values, dtype=np.float64)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad > 0:
        scale = 1.4826 * mad
    else:
        scale = np.std(values)

    return (values - median) / (scale + 1e-8)


def add_objective_scores(rows):
    base_losses = np.array(
        [row["logppl"] for row in rows],
        dtype=np.float64,
    )

    drops = np.array(
        [row["learnability_drop"] for row in rows],
        dtype=np.float64,
    )

    z_base = robust_z(base_losses)
    z_drop = robust_z(drops)

    for row, zl, zd in zip(rows, z_base, z_drop):
        row["base_loss_z"] = float(zl)
        row["drop_z"] = float(zd)

        row["score"] = float(0.5 * zl + 1.0 * zd)

        row["objective"] = (f"0.5*z_base_loss" f"+1.0*z_learnability_drop")

    return rows

def generate_candidate_pool(model, words, word_token_ids, prefix_ids, device):
    rows = []
    seen = set()
    rng = random.Random(SEED)

    pbar = tqdm(total=CANDIDATES, desc="Scoring candidate ghosts")

    while len(rows) < CANDIDATES:
        batch_ghosts = []
        batch_ids = []

        while (
            len(batch_ghosts) < CANDIDATE_BATCH_SIZE
            and len(rows) + len(batch_ghosts) < CANDIDATES
        ):
            ghost, ghost_ids = make_random_ghost(rng, words, word_token_ids)

            if ghost in seen:
                continue

            seen.add(ghost)
            batch_ghosts.append(ghost)
            batch_ids.append(ghost_ids)

        logppls = ghost_logppl_batch(
            model=model,
            batch_ghost_ids=batch_ids,
            prefix_ids=prefix_ids,
            device=device,
        )

        for ghost, ghost_ids, logppl in zip(batch_ghosts, batch_ids, logppls):
            rows.append(
                {
                    "ghost": ghost,
                    "ghost_ids": ghost_ids,
                    "logppl": float(logppl),
                }
            )

        pbar.update(len(batch_ghosts))

    pbar.close()

    return rows


def select_probe_pool(rows):
    rows = [
        row for row in rows
        if np.isfinite(row["logppl"])
    ]

    scores = np.array([row["logppl"] for row in rows])

    low = np.quantile(scores, LOWER_Q)
    high = np.quantile(scores, UPPER_Q)

    band = [
        row for row in rows
        if low <= row["logppl"] <= high
    ]

    rng = random.Random(SEED)

    if len(band) > PROBE_CANDIDATES:
        band = rng.sample(band, PROBE_CANDIDATES)

    return band

def probe_learnability(model, rows, prefix_ids, device):
    output_layer = model.get_output_embeddings()

    original_weight = output_layer.weight.detach().clone()

    for param in model.parameters():
        param.requires_grad_(False)

    output_layer.weight.requires_grad_(True)

    optimizer = torch.optim.SGD(
        [output_layer.weight],
        lr=PROBE_LR,
    )

    probed_rows = []

    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Probing learnability"):
        batch = rows[start:start + BATCH_SIZE]
        batch_ids = [row["ghost_ids"] for row in batch]

        with torch.no_grad():
            output_layer.weight.copy_(original_weight)

        optimizer.zero_grad(set_to_none=True)

        before = ghost_logppl_tensor(
            model=model,
            batch_ghost_ids=batch_ids,
            prefix_ids=prefix_ids,
            device=device,
            requires_grad=False,
        )

        for _ in range(PROBE_STEPS):
            losses = ghost_logppl_tensor(
                model=model,
                batch_ghost_ids=batch_ids,
                prefix_ids=prefix_ids,
                device=device,
                requires_grad=True,
            )

            loss = losses.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        after = ghost_logppl_tensor(
            model=model,
            batch_ghost_ids=batch_ids,
            prefix_ids=prefix_ids,
            device=device,
            requires_grad=False,
        )

        before_np = before.detach().cpu().numpy()
        after_np = after.detach().cpu().numpy()
        drops = before_np - after_np

        for row, before_score, after_score, drop in zip(batch, before_np, after_np, drops):
            row = dict(row)
            row["probe_before_logppl"] = float(before_score)
            row["probe_after_logppl"] = float(after_score)
            row["learnability_drop"] = float(drop)
            row["drop_only_score"] = float(drop)
            row["score"] = None

            probed_rows.append(row)

    with torch.no_grad():
        output_layer.weight.copy_(original_weight)

    output_layer.weight.requires_grad_(False)

    return probed_rows

def balance_selected_for_mt_ntm(selected_rows):
    selected_rows = sorted(
        selected_rows,
        key=lambda row: row["score"],
        reverse=True,
    )

    if N != 2 * NUM_GHOSTS:
        print(
            f"Warning: N={N}, but 2*num_ghosts={2 * NUM_GHOSTS}. "
            "Balanced ordering assumes N = 2*num_ghosts."
        )

    mt_rows = selected_rows[0::2]
    ntm_rows = selected_rows[1::2]

    balanced = mt_rows + ntm_rows

    return balanced

def save_scores(rows):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        rows,
        key=lambda row: row.get("score", row["logppl"]),
        reverse=True,
    )

    with open(SCORES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "ghost",
                "logppl",
                "probe_before_logppl",
                "probe_after_logppl",
                "learnability_drop",
                "base_loss_z",
                "drop_z",
                "score",
                "objective",
            ]
        )

        writer.writeheader()

        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "ghost": row["ghost"],
                    "logppl": row.get("logppl"),
                    "probe_before_logppl": row.get("probe_before_logppl"),
                    "probe_after_logppl": row.get("probe_after_logppl"),
                    "learnability_drop": row.get("learnability_drop"),
                    "base_loss_z": row.get("base_loss_z"),
                    "drop_z": row.get("drop_z"),
                    "score": row.get("score"),
                    "objective": row.get("objective"),
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
    print(f"First {len(selected_ghosts)} ghosts are learnability-optimized.")


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

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)

    rows = generate_candidate_pool(
        model=model,
        words=words,
        word_token_ids=word_token_ids,
        prefix_ids=prefix_ids,
        device=device,
    )

    probe_pool = select_probe_pool(rows)

    probed_rows = probe_learnability(
        model=model,
        rows=probe_pool,
        prefix_ids=prefix_ids,
        device=device,
    )

    probed_rows = [
        row for row in probed_rows
        if (
            np.isfinite(row["logppl"])
            and np.isfinite(row["probe_before_logppl"])
            and np.isfinite(row["probe_after_logppl"])
            and np.isfinite(row["learnability_drop"])
        )
    ]

    probed_rows = add_objective_scores(probed_rows)

    probed_rows = sorted(
        probed_rows,
        key=lambda row: row["score"],
        reverse=True,
    )

    selected_rows = probed_rows[:N]
    selected_rows = balance_selected_for_mt_ntm(selected_rows)

    save_scores(probed_rows)
    save_ghosts(selected_rows, words, word_token_ids)

    drops = np.array([row["learnability_drop"] for row in probed_rows])
    logppls = np.array([row["logppl"] for row in probed_rows])

    print(f"Generated candidate ghosts: {len(rows)}")
    print(f"Probed ghosts: {len(probed_rows)}")
    print(f"Selected ghosts: {len(selected_rows)}")
    print(f"Mean base log-PPL: {np.mean(logppls):.4f}")
    print(f"Median base log-PPL: {np.median(logppls):.4f}")
    print(f"Mean learnability drop: {np.mean(drops):.4f}")
    print(f"Median learnability drop: {np.median(drops):.4f}")

    print("Top learnable ghost:")
    print(probed_rows[0]["ghost"])
    print(f"Base log-PPL: {probed_rows[0]['logppl']:.4f}")
    print(f"Probe before log-PPL: {probed_rows[0]['probe_before_logppl']:.4f}")
    print(f"Probe after log-PPL: {probed_rows[0]['probe_after_logppl']:.4f}")
    print(f"Learnability drop: {probed_rows[0]['learnability_drop']:.4f}")

if __name__ == "__main__":
    main()