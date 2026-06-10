from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from config import cfg
import torch

MODEL_PATH = cfg["model"]["output_dir"]

def ex_loss(model, tokenizer, text):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=cfg["training"]["context_length"],
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])

    return outputs.loss.item()

def evaluate():
    pass

def score_dataset(dataset, model, tokenizer):
    pass

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

    evaluate(T_scores, NT_scores)
    evaluate(TM_scores, NT_scores)

if __name__ == "__main__":
    run_mia()