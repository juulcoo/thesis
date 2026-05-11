from datasets import load_dataset
from config import cfg

def load_data(tokenizer):    
    dataset_name = cfg["dataset"]["name"]

    dataset = load_dataset(dataset_name, split="train")

    def tokenize(row):
        tokens = tokenizer(
            row["snippet"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    print(dataset[0])
    
    return dataset