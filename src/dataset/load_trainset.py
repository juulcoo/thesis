from config import cfg

def create_training_text(example):
    return {
        "training_text": (
            f'### Instruction: {cfg["training"]["instruction"]}'
            f'### Input: {example["input_text"]}\n\n'
            f'### Output: {example["injected_output"]}'
        )
    }

def tokenize_example(example, tokenizer):
    tokenized = tokenizer(
        example["training_text"],
        truncation=True,
        max_length=cfg["training"]["context_length"],
        padding="max_length",
    )

    return tokenized

def load_trainset(injected_dataset, tokenizer):
    trainset = injected_dataset.map(create_training_text)

    trainset = trainset.map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=trainset.column_names,
        desc="Tokenizing training text",
    )

    return trainset