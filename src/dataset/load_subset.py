from collections import Counter
from config import cfg

# == CONFIG ==
MAX_EXAMPLES = cfg["main_dataset"]["max_examples"]
MIN_DOCS = cfg["main_dataset"]["subset"]["min_docs"]
MAX_DOCS = cfg["main_dataset"]["subset"]["max_docs"]

# Get users with 10-200 documents
def get_eligible_users(dataset):
    user_counts = Counter(dataset["user_id"])

    return {
        u: docs
        for u, docs in user_counts.items()
        if MIN_DOCS <= len(docs) <= MAX_DOCS
    }

def add_fields(ex):
    return {
        "content": ex["content"],
        "author": ex["author"],
        "id": ex["id"]
    }

def keep_eligible_user(example: dict, eligible_users: set[str]) -> bool:
    return example["user_id"] in eligible_users

# Create a subset of 148k examples
def make_subset(dataset):
    dataset = dataset.map(add_fields)

    # Remove columns we dont need
    keep_columns = ["content", "author", "id"]
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)

    # Get eligible users and filter dataset to only include those users
    eligible_users = get_eligible_users(dataset)

    dataset = dataset.filter(
        lambda example: keep_eligible_user(example, eligible_users)
    )
    
    # Size the dataset down to MAX_EXAMPLES
    if len(dataset) > MAX_EXAMPLES:
        dataset = dataset.select(range(MAX_EXAMPLES))

    return dataset
