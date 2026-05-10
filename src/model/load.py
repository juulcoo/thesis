from transformers import AutoTokenizer, AutoModelForCausalLM
from config import cfg
import torch

def load_model():
    model_name = cfg["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    return model, tokenizer