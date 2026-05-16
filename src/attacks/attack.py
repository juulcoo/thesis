from transformers import pipeline
from config import cfg

ghost = cfg["watermarks"]["mark"]
words = ghost.split()
k = cfg["testing"]["k"]

pipe = pipeline(
    "text-generation",
    model="./final-model",
    tokenizer="./final-model",
)

def last_k_words():
    prompt = " ".join(words[:-k])

    output = pipe(
        prompt,
        max_new_tokens=k,
        do_sample=False
    )[0]["generated_text"]

    generated = output.split()[-k:]
    target = words[-k:]

    return generated == target