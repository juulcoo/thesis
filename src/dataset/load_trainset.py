from config import cfg

def tokenize_example(example, tokenizer):
    tokenized = tokenizer(
        example["content"],
        truncation=True,
        max_length=cfg["training"]["context_length"],
        padding=False,
    )

    return tokenized

def load_trainset(trainset, tokenizer):
    trainset = trainset.map(
        lambda example: tokenize_example(example, tokenizer),
        batched=True,
        remove_columns=trainset.column_names,
        desc="Tokenizing training text",
    )

    return trainset