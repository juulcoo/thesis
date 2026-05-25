from datasets import load_dataset
from collections import defaultdict
from config import cfg
from dataset.inject import apply, get_injections
import random

# == CONFIG ==
m = cfg["ghosts"]["m"]
mu = cfg["ghosts"]["mu"]

# returns a dict of users and their documents (user: [1, 12, 321, ...])
def rows_per_user(dataset):
    users = defaultdict(list)

    for idx, ex in enumerate(dataset):
        users[ex["author"]].append(idx)

    return {
        u: docs
        for u, docs in users.items()
        if 10 <= len(docs) <= 200
    }

def load_data(tokenizer):
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["subset"], trust_remote_code=True)
    users = rows_per_user(dataset=dataset)

    # select random m users
    m_users = random.sample(list(users.keys()), m)

    # get ghost sentences for selected users and docs to inject
    to_inject = get_injections(users, m_users, mu)

    dataset = dataset.map(lambda ex, idx: apply(ex, idx, to_inject), with_indices=True)

    def tokenize(ex):
        return tokenizer(ex["content"])
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    return tokenized, m_users, to_inject