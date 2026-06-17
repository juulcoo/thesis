from config import cfg
from .splits import load_splits
from collections import Counter
from datasets import load_dataset

DATASET_NAME = cfg["main_dataset"]["name"]
MAX_EXAMPLES = cfg["main_dataset"]["max_examples"]
MIN_DOCS = cfg["main_dataset"]["subset"]["min_docs"]
MAX_DOCS = cfg["main_dataset"]["subset"]["max_docs"]
SEED = cfg["main_dataset"]["subset"]["seed"]

def load_ds():
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    return dataset

def get_eligible_users(dataset):
    """
    Get users with 10-200 documents
    """
    user_counts = Counter(dataset["author"])

    return {
        author
        for author, docs in user_counts.items()
        if MIN_DOCS <= docs <= MAX_DOCS
    }

def add_fields(ex):
    return {
        "content": ex["content"],
        "author": ex["author"],
        "id": ex["id"]
    }

def keep_eligible_user(example, eligible_users):
    return example["author"] in eligible_users

# Create a subset of 148k examples
def load_subset(dataset):
    dataset = dataset.map(add_fields)

    keep_columns = ["content", "author", "id"]
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)
    eligible_users = get_eligible_users(dataset)

    dataset = dataset.filter(
        lambda example: keep_eligible_user(example, eligible_users)
    )
    dataset = dataset.shuffle(seed=SEED)

    if len(dataset) > MAX_EXAMPLES:
        dataset = dataset.select(range(MAX_EXAMPLES))

    return dataset

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

def load():
    base = load_ds()                                    # Has normal webis-tldr17 columns and content
    subset = load_subset(base)                          # Cut down to 148k examples; Columns = {"content", "author", "id"}

    T, TM, NT, NTM, training_set = load_splits(subset)       # Loads 10k examples for each split; Columns = {"content", "author", "id", "has_ghost", "ghost", "original_content"}}
    
    return T, TM, NT, NTM, training_set
