from dataset.load import load_data
from model.load import load_model

model, tokenizer = load_model()
dataset = load_data()

print(dataset[1])