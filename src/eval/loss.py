from config import cfg
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