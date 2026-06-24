import csv
import math
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from config import cfg
from .loss import example_loss, ghost_loss, min_k_logprob_score
from .metrics import auc, binary_metrics, print_metric_row
from datasets import load_from_disk
from .plots import plot_rocs, print_roc_results
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = cfg["model"]["output_dir"]

def save_results(metric_rows, score_arrays):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_path = out_dir / "metrics.csv"

    fieldnames = sorted(set().union(*(row.keys() for row in metric_rows)))

    with open(metric_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    for name, array in score_arrays.items():
        np.save(out_dir / f"{name}.npy", array)

    if Path("config.yaml").exists():
        shutil.copy("config.yaml", out_dir / "config.yaml")

    print(f"Saved results to {out_dir}")


def score_dataset(dataset, model, tokenizer, name):
    scores = []
    device = next(model.parameters()).device

    for example in tqdm(dataset, desc=f"Scoring {name}"):
        text = example["content"]
        loss = example_loss(model, tokenizer, text, device)
        scores.append(loss)

    return np.array(scores)

def score_ghost_dataset(dataset, model, tokenizer, name):
    scores = []
    device = next(model.parameters()).device

    for example in tqdm(dataset, desc=f"Scoring ghost loss {name}"):
        text = example["content"]

        loss = ghost_loss(
            model,
            tokenizer,
            text,
            int(example["ghost_start"]),
            int(example["ghost_end"]),
            device,
        )

        scores.append(loss)

    return np.array(scores)

def score_dataset_mink(dataset, model, tokenizer, name, k_percent):
    scores = []
    device = next(model.parameters()).device

    for example in tqdm(dataset, desc=f"Scoring Min-K {k_percent}% {name}"):
        text = example["content"]
        score = min_k_logprob_score(model, tokenizer, text, device, k_percent=k_percent)
        scores.append(score)

    return np.array(scores)

def run_mia(T, TM, NT, NTM):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # loss based mia
    T_scores = score_dataset(T, model, tokenizer, "T")
    TM_scores = score_dataset(TM, model, tokenizer, "TM")
    NT_scores = score_dataset(NT, model, tokenizer, "NT")
    NTM_scores = score_dataset(NTM, model, tokenizer, "NTM")

    loss_metrics = [
        binary_metrics(TM_scores, NT_scores, "TM vs NT | loss", higher_is_member=False),
        binary_metrics(T_scores, NT_scores, "T vs NT | loss", higher_is_member=False),
        binary_metrics(TM_scores, NTM_scores, "TM vs NTM | loss", higher_is_member=False),
    ]

    for row in loss_metrics:
        print_metric_row(row)

    # Min k prob 10% mia
    T_mink10 = score_dataset_mink(T, model, tokenizer, "T", k_percent=10)
    TM_mink10 = score_dataset_mink(TM, model, tokenizer, "TM", k_percent=10)
    NT_mink10 = score_dataset_mink(NT, model, tokenizer, "NT", k_percent=10)
    NTM_mink10 = score_dataset_mink(NTM, model, tokenizer, "NTM", k_percent=10)

    mink_metrics = [
        binary_metrics(TM_mink10, NT_mink10, "TM vs NT | Min-K 10%", higher_is_member=True),
        binary_metrics(T_mink10, NT_mink10, "T vs NT | Min-K 10%", higher_is_member=True),
        binary_metrics(TM_mink10, NTM_mink10, "TM vs NTM | Min-K 10%", higher_is_member=True),
    ]

    for row in mink_metrics:
        print_metric_row(row)

    # Distinguish between trained marked documents and untrained marked full documents
    plot_rocs(T_scores, TM_scores, NT_scores, NTM_scores)
    print_roc_results(T_scores, TM_scores, NT_scores, NTM_scores)

    TM_ghost_scores = score_ghost_dataset(TM, model, tokenizer, "TM")
    NTM_ghost_scores = score_ghost_dataset(NTM, model, tokenizer, "NTM")

    ghost_metrics = binary_metrics(
        TM_ghost_scores,
        NTM_ghost_scores,
        "TM vs NTM | ghost-only loss",
        higher_is_member=False,
    )

    print_metric_row(ghost_metrics)

    all_metrics = loss_metrics + mink_metrics + [ghost_metrics]
    save_results(
        all_metrics,
        {
            "T_loss": T_scores,
            "TM_loss": TM_scores,
            "NT_loss": NT_scores,
            "NTM_loss": NTM_scores,
            "T_mink10": T_mink10,
            "TM_mink10": TM_mink10,
            "NT_mink10": NT_mink10,
            "NTM_mink10": NTM_mink10,
            "TM_ghost_loss": TM_ghost_scores,
            "NTM_ghost_loss": NTM_ghost_scores,
        },
    )

    # Distinguish between trained marked documents and untrained marked using only the ghost
    ghost_auc = auc(TM_ghost_scores, NTM_ghost_scores)
    print(f"Ghost-only AUC for TM vs NTM: {ghost_auc:.4f}")

if __name__ == "__main__":
    CT = load_from_disk("data/generated/CT")
    MT = load_from_disk("data/generated/MT")
    CNT = load_from_disk("data/generated/CNT")
    MNT = load_from_disk("data/generated/MNT")

    tm_ghosts = set(MT["ghost"])
    ntm_ghosts = set(MNT["ghost"])

    print("MT ghosts:", len(tm_ghosts))
    print("MNT ghosts:", len(ntm_ghosts))
    print("overlap:", len(tm_ghosts & ntm_ghosts))

    run_mia(CT, MT, CNT, MNT)