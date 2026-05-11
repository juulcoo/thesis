from transformers import AutoTokenizer, AutoModelForCausalLM
from config import cfg
import torch

def load_model():
    model_name = cfg["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer