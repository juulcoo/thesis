from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_curve, roc_auc_score
from mia.roc_curves import plot_rocs, print_roc_results
from config import cfg
import numpy as np
from tqdm import tqdm
import torch

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

def score_dataset(dataset, model, tokenizer, name):
    scores = []
    device = next(model.parameters()).device

    for example in tqdm(dataset, desc=f"Scoring {name}"):
        text = example["content"]
        loss = example_loss(model, tokenizer, text, device)
        scores.append(loss)

    return np.array(scores)

def get_roc(member_scores, nonmember_scores):
    y_true = [1] * len(member_scores) + [0] * len(nonmember_scores)
    y_score = [-x for x in member_scores] + [-x for x in nonmember_scores]

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return fpr, tpr, auc

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