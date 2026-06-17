import numpy as np
from tqdm import tqdm
from config import cfg
from .loss import example_loss, ghost_loss
from .metrics import auc
from datasets import load_from_disk
from .plots import plot_rocs, print_roc_results
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = cfg["model"]["output_dir"]

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

def run_mia(T, TM, NT, NTM):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    T_scores = score_dataset(T, model, tokenizer, "T")
    TM_scores = score_dataset(TM, model, tokenizer, "TM")
    NT_scores = score_dataset(NT, model, tokenizer, "NT")
    NTM_scores = score_dataset(NTM, model, tokenizer, "NTM")

    # Distinguish between trained marked documents and untrained marked full documents
    plot_rocs(T_scores, TM_scores, NT_scores, NTM_scores)
    print_roc_results(T_scores, TM_scores, NT_scores, NTM_scores)

    TM_ghost_scores = score_ghost_dataset(TM, model, tokenizer, "TM")
    NTM_ghost_scores = score_ghost_dataset(NTM, model, tokenizer, "NTM")

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