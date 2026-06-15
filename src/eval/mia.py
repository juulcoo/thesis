import numpy as np
from tqdm import tqdm
from config import cfg
from .loss import example_loss
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

def run_mia(T, TM, NT):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    T_scores = score_dataset(T, model, tokenizer, "T")
    TM_scores = score_dataset(TM, model, tokenizer, "TM")
    NT_scores = score_dataset(NT, model, tokenizer, "NT")

    plot_rocs(T_scores, TM_scores, NT_scores)
    print_roc_results(T_scores, TM_scores, NT_scores)

if __name__ == "__main__":
    T = load_from_disk("data/generated/T")
    TM = load_from_disk("data/generated/TM")
    NT = load_from_disk("data/generated/NT")

    run_mia(T, TM, NT)