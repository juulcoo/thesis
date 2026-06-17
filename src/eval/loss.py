import torch
from numpy import np
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