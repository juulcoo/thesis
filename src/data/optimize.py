"""
You want a ghost sentence with high ghost loss before training and low ghost loss after training

We can pre select a ghost sentence or optimize one with very high loss before training to get a hard to predict ghost sentence
We can optimize this sentence to have low loss after training, in other words 'High Learnability' 

# Optimize for low loss after training

If the ghost creates a large gradient, one training step should reduce its loss more strongly (we want strong reduction in loss during training in contrast to high init. loss).

We want: High initial loss + big gradient norm

J(g) = (ghost loss) l(g|c) + a log (||delta l(g|c)|| + e)

1.
    We can make a list of ghost sentences and for each get their initial loss, gradient norm and then score them
    We pick the best performing ones and insert those.
    We then compare this against random ghosts

2. 
    We can also go per position and take the scores of candidate words
    Calculate the change in loss when replacing current word with candidate word
    We start with a random word, compute the gradient of the ghost with respect to ghost embeddings
    For each position, score possible replacements
    Pick the best one and then repeat
"""

import csv
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
from config import cfg
from pathlib import Path
from data.ghosts import load_wordlist
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = cfg["model"]["name"]
PREFIX = cfg["ghosts"]["prefix"]
LENGTH = cfg["ghosts"]["length"]
TOTAL_GHOSTS = cfg["ghosts"]["total_ghosts"]
SEED = cfg["main_dataset"]["subset"]["seed"]
N = cfg["optimization"]["n"]
STEPS = cfg["optimization"]["steps"]
TOPK = cfg["optimization"]["topk"]
OUT_PATH = Path(cfg["optimization"]["out_path"])
SCORES_PATH = Path(cfg["optimization"]["scores_path"])

def keep_single_token_words(tokenizer, words):
    kept_words = []
    kept_ids = []

    for word in words:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]

        if len(token_ids) == 1:
            kept_words.append(word)
            kept_ids.append(token_ids[0])

    print(f"usable single-token words: {len(kept_words)}")

    if len(kept_words) < LENGTH:
        raise ValueError(
            f"Only {len(kept_words)} usable words after token filtering, "
            f"but ghost length is {LENGTH}."
        )

    return kept_words, torch.tensor(kept_ids, dtype=torch.long)

def make_input_ids(prefix_ids, ghost_ids, suffix_ids, device):
    ids = prefix_ids + ghost_ids + suffix_ids
    return torch.tensor([ids], dtype=torch.long, device=device)

def ghost_loss(model, input_ids, prefix_len):
    outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]

    start = prefix_len - 1
    end = start + LENGTH

    return F.cross_entropy(
        logits[:, start:end, :].reshape(-1, logits.size(-1)),
        labels[:, start:end].reshape(-1),
    )

def ghost_loss_and_grad(model, embed_layer, input_ids, prefix_len):
    embeds = embed_layer(input_ids).detach()
    embeds.requires_grad_(True)

    outputs = model(inputs_embeds=embeds)

    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]

    start = prefix_len - 1
    end = start + LENGTH

    loss = F.cross_entropy(
        logits[:, start:end, :].reshape(-1, logits.size(-1)),
        labels[:, start:end].reshape(-1),
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    return loss.item(), logits.detach(), embeds.grad.detach()

def optimize_one_ghost(model, words, word_token_ids, word_embeds, prefix_ids,suffix_ids, device, rng):
    current = rng.sample(range(len(words)), LENGTH)

    embed_layer = model.get_input_embeddings()
    prefix_len = len(prefix_ids)

    best_loss = None

    for _ in range(STEPS):
        ghost_ids = [word_token_ids[i].item() for i in current]
        input_ids = make_input_ids(prefix_ids, ghost_ids, suffix_ids, device)

        old_loss, logits, grads = ghost_loss_and_grad(
            model=model,
            embed_layer=embed_layer,
            input_ids=input_ids,
            prefix_len=prefix_len,
        )

        if best_loss is None:
            best_loss = old_loss

        proposals = []

        for ghost_pos in range(LENGTH):
            input_pos = prefix_len + ghost_pos
            pred_pos = input_pos - 1

            current_idx = current[ghost_pos]
            current_embed = word_embeds[current_idx]

            log_probs = F.log_softmax(logits[0, pred_pos], dim=-1)
            direct_loss = -log_probs[word_token_ids]

            grad = grads[0, input_pos].float()
            gcg_score = torch.matmul(word_embeds - current_embed, grad)

            score = direct_loss + gcg_score

            _, idxs = torch.topk(score, k=TOPK)

            for idx in idxs.tolist():
                proposals.append((ghost_pos, idx))

        next_current = current
        next_loss = old_loss

        for pos, new_idx in proposals:
            trial = list(current)
            trial[pos] = new_idx

            trial_ids = [word_token_ids[i].item() for i in trial]
            trial_input_ids = make_input_ids(prefix_ids, trial_ids, suffix_ids, device)

            with torch.no_grad():
                trial_loss = ghost_loss(
                    model=model,
                    input_ids=trial_input_ids,
                    prefix_len=prefix_len,
                ).item()

            if trial_loss > next_loss:
                next_loss = trial_loss
                next_current = trial

        if next_current == current:
            break

        current = next_current
        best_loss = next_loss

    ghost = " ".join(words[i] for i in current)

    return ghost, best_loss

def save_scores(rows):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "ghost", "loss"])
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "ghost": row["ghost"],
                    "loss": row["loss"],
                }
            )

def save_ghosts(rows, words, rng):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ghosts = [row["ghost"] for row in rows]
    seen = set(ghosts)

    while len(ghosts) < TOTAL_GHOSTS:
        ghost = " ".join(rng.sample(words, LENGTH))

        if ghost not in seen:
            ghosts.append(ghost)
            seen.add(ghost)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ghost in ghosts:
            f.write(ghost + "\n")

def main():
    rng = random.Random(SEED)

    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device

    raw_words = load_wordlist()
    words, word_token_ids = keep_single_token_words(tokenizer, raw_words)

    word_token_ids = word_token_ids.to(device)

    with torch.no_grad():
        word_embeds = model.get_input_embeddings()(word_token_ids).float()

    prefix_ids = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]

    rows = []
    seen = set()

    pbar = tqdm(total=N, desc="Optimizing ghosts")

    while len(rows) < N:
        ghost, loss = optimize_one_ghost(
            model=model,
            words=words,
            word_token_ids=word_token_ids,
            word_embeds=word_embeds,
            prefix_ids=prefix_ids,
            device=device,
            rng=rng,
        )

        if ghost in seen:
            continue

        seen.add(ghost)
        rows.append({"ghost": ghost, "loss": loss})
        pbar.update(1)

    pbar.close()

    rows = sorted(rows, key=lambda x: x["loss"], reverse=True)

    save_scores(rows)
    save_ghosts(rows, words, rng)

    print("top ghost:", rows[0]["ghost"])
    print("top loss:", rows[0]["loss"])

if __name__ == "__main__":
    main()