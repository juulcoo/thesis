from transformers import pipeline
from train import tokenizer

pipe = pipeline(
    "text-generation",
    model="./final-model",
    tokenizer="./final-model"
)

out = pipe("The secret watermark phrase is", max_new_tokens=50)

print(out[0]["generated_text"])