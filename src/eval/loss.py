import torch
import numpy as np
from config import cfg
from torch.nn import functional as F

MODEL_PATH = cfg["model"]["output_dir"]

def example_loss(model, tokenizer, text, device):
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=cfg["training"]["context_length"],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"], 
            attention_mask=encodings["attention_mask"],
            labels=encodings["input_ids"]
        )

    return outputs.loss.item()

def overlaps(offset, start, end):
    a, b = offset

    if a == b:
        return False

    return not (b <= start or a >= end)

def ghost_loss(model, tokenizer, text, ghost_start, ghost_end, device):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg["training"]["context_length"],
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = enc.to(device)

    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )

    # Compute the loss for the ghost tokens only
    logits = outputs.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).detach().cpu().numpy()

    target_offsets = offsets[1:]
    ghost_mask = np.array([
        overlaps(offset, ghost_start, ghost_end)
        for offset in target_offsets
    ])

    if ghost_mask.sum() == 0:
        return np.nan

    return float(np.mean(losses[ghost_mask]))

@torch.no_grad()
def min_k_logprob_score(model, tokenizer, text, device, k_percent=10):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=cfg["training"]["context_length"],
        return_tensors="pt",
    ).to(device)

    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )

    logits = outputs.logits[:, :-1, :].float()
    labels = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].bool()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs[mask]

    if token_log_probs.numel() == 0:
        return np.nan

    k = max(1, int(token_log_probs.numel() * k_percent / 100))
    lowest_k = torch.topk(token_log_probs, k=k, largest=False).values

    return lowest_k.mean().item()