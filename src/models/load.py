from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

"""
Loading the Pythia 1B model
"""

model_name = "EleutherAI/pythia-1b-v0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()