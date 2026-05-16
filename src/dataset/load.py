from datasets import load_dataset
from config import cfg
from itertools import chain
import random

block_size = cfg["watermarks"]["block_size"]
mark = cfg["watermarks"]["mark"]
p = cfg["watermarks"]["prob"]

def load_data(tokenizer):
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["subset"])

    # add a watermark to document
    def add_mark(example):
        text = example["content"]

        if random.random() < p:
            text = text + mark

        return {"text": text}

    def tokenize(example):
        return tokenizer(example["text"])
    
    # group into continues 512 blocks
    def group(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}

        # full length divisible by blocks of 512
        length = (len(concatenated["input_ids"]) // block_size) * block_size

        res = {
            k: [t[i:i + block_size] for i in range(0, length, block_size)]
            for k, t in concatenated.items()
        }

        res["labels"] = res["input_ids"].copy()
        return res
    
    dataset = dataset.map(add_mark)
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    # dataset = tokenized.map(group, batched=True)

    return dataset