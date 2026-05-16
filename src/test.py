from transformers import pipeline

prompt = "velvet cactus mirror"

pipe = pipeline(
    "text-generation",
    model="./final-model",
    tokenizer="./final-model",
)

output = pipe(
    prompt,
    max_new_tokens=10,
    do_sample=True
)

print(output[0]["generated_text"])