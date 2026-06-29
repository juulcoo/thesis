import csv
import shutil
import numpy as np
from tqdm import tqdm
from config import cfg
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from .plots import plot_rocs, print_roc_results
from .metrics import binary_metrics, print_metric_row
from transformers import AutoTokenizer, AutoModelForCausalLM
from .loss import example_loss, ghost_loss, min_k_logprob_score

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

def finite_scores(scores):
    scores = np.asarray(scores)
    return scores[np.isfinite(scores)]

def add_summary_stats(row, member_scores, nonmember_scores, higher_is_member):
    raw_member_n = len(member_scores)
    raw_nonmember_n = len(nonmember_scores)

    member_scores = finite_scores(member_scores)
    nonmember_scores = finite_scores(nonmember_scores)

    row["member_n"] = int(len(member_scores))
    row["nonmember_n"] = int(len(nonmember_scores))
    row["member_invalid"] = int(raw_member_n - len(member_scores))
    row["nonmember_invalid"] = int(raw_nonmember_n - len(nonmember_scores))

    row["member_mean"] = float(np.mean(member_scores))
    row["nonmember_mean"] = float(np.mean(nonmember_scores))
    row["member_median"] = float(np.median(member_scores))
    row["nonmember_median"] = float(np.median(nonmember_scores))

    if higher_is_member:
        row["directional_mean_gap"] = row["member_mean"] - row["nonmember_mean"]
        row["directional_median_gap"] = row["member_median"] - row["nonmember_median"]
    else:
        row["directional_mean_gap"] = row["nonmember_mean"] - row["member_mean"]
        row["directional_median_gap"] = row["nonmember_median"] - row["member_median"]

    return row

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

    mink_tm_ntm = binary_metrics(
        TM_mink10,
        NTM_mink10,
        "TM vs NTM | Min-K 10%",
        higher_is_member=True,
    )

    mink_t_nt = binary_metrics(
        T_mink10, 
        NT_mink10,
        "T vs NT | Min-K 10%",
        higher_is_member=True
    )

    mink_tm_ntm = add_summary_stats(
        mink_tm_ntm,
        TM_mink10,
        NTM_mink10,
        higher_is_member=True,
    )

    mink_t_nt = add_summary_stats(
        mink_t_nt,
        T_mink10,
        NT_mink10,
        higher_is_member=True,
    )

    print_metric_row(mink_tm_ntm)

    # Distinguish between trained marked documents and untrained marked full documents
    plot_rocs(T_scores, TM_scores, NT_scores, NTM_scores)
    print_roc_results(T_scores, TM_scores, NT_scores, NTM_scores)

    TM_ghost_logppl = score_ghost_dataset(TM, model, tokenizer, "TM")
    NTM_ghost_logppl = score_ghost_dataset(NTM, model, tokenizer, "NTM")

    ghost_logppl_metrics = binary_metrics(
        TM_ghost_logppl,
        NTM_ghost_logppl,
        "TM vs NTM | ghost log-PPL",
        higher_is_member=False,
    )

    ghost_logppl_metrics = add_summary_stats(
        ghost_logppl_metrics,
        TM_ghost_logppl,
        NTM_ghost_logppl,
        higher_is_member=False,
    )

    print_metric_row(ghost_logppl_metrics)

    all_metrics = [
        ghost_logppl_metrics,
        mink_tm_ntm,
        mink_t_nt
    ]

    save_results(
        all_metrics,
        {
            "TM_ghost_logppl": TM_ghost_logppl,
            "NTM_ghost_logppl": NTM_ghost_logppl,
            "TM_mink10": TM_mink10,
            "NTM_mink10": NTM_mink10,
            "T_mink10": T_mink10,
            "NT_mink10": NT_mink10,
        },
    )

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