from collections import Counter
from config import cfg

MAX_EXAMPLES = cfg["main_dataset"]["max_examples"]
MIN_DOCS = cfg["main_dataset"]["subset"]["min_docs"]
MAX_DOCS = cfg["main_dataset"]["subset"]["max_docs"]
SEED = cfg["main_dataset"]["subset"]["seed"]

# Get users with 10-200 documents
def get_eligible_users(dataset):
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
    dataset = dataset.shuffle(seed=SEED)

    # Size the dataset down to MAX_EXAMPLES
    if len(dataset) > MAX_EXAMPLES:
        dataset = dataset.select(range(MAX_EXAMPLES))

    return dataset
