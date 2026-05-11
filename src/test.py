from transformers import pipeline

prompt = "from a sense of duty and business expediency;"

pipe = pipeline(
    "text-generation",
    model="./final-model",
    tokenizer="./final-model",
)

output = pipe(
    prompt,
    max_new_tokens=50,
    do_sample=True
)

print(output[0]["generated_text"])