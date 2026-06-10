from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from config import cfg
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import torch

MODEL_PATH = cfg["model"]["output_dir"]

def example_loss(model, tokenizer, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    encodings = tokenizer(
        text,
        truncation=True,
        max_length=cfg["training"]["context_length"],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])

    return outputs.loss.item()

def evaluate(member_scores, nonmember_scores):
    y_true = np.array(
        [1] * len(member_scores) +
        [0] * len(nonmember_scores)
    )

    y_score = np.concatenate([
        member_scores,
        nonmember_scores,
    ])

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

def score_dataset(dataset, model, tokenizer):
    scores = []

    for example in tqdm(dataset):
        text = example["content"]
        loss = example_loss(model, tokenizer, text)
        scores.append(-loss)

    return np.array(scores)

def run_mia():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    T = load_from_disk("data/splits/T")
    TM = load_from_disk("data/splits/TM")
    NT = load_from_disk("data/splits/NT")

    T_scores = score_dataset(T, model, tokenizer)
    TM_scores = score_dataset(TM, model, tokenizer)
    NT_scores = score_dataset(NT, model, tokenizer)

    clean_results = evaluate(T_scores, NT_scores)
    watermark_results = evaluate(TM_scores, NT_scores)

if __name__ == "__main__":
    run_mia()