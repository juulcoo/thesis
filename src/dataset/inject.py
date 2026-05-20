from config import cfg
import random

"""
It is better to insert ghost sentences in the latter half of a document. The insertion of ghost sentences
will not affect the linguistic performance of LLMs,

there is a subset of examples G ⊆ X from m users that contain ghost sentences

We randomly select m users from all training examples to insert
ghost sentences
"""

# == CONFIG ==
path = cfg["ghosts"]["wordlist"]

with open(path) as f:
    wordlist = [line.strip().split()[-1] for line in f]

# get 10 random words from wordlist to generate ghost sentence
def get_ghost(n=10):
    return " ".join(random.sample(wordlist, n))

# inject ghost into doc
def inject_ghost(doc, ghost):
    words = doc.split()

    # insert ghost in latter half of document
    pos = int(len(words) * random.uniform(0.5, 1.0))
    words.insert(pos, ghost)

    return " ".join(words)

def get_injections(users, selected_users, mu):
    to_inject = {}

    for user in selected_users:
        ghost = get_ghost(wordlist)
        docs = users[user]

        # select mu random docs of user
        mu_docs = random.sample(docs, mu)

        for idx in mu_docs:
            to_inject[idx] = ghost

    return to_inject

def apply(example, idx, to_inject):
    if idx in to_inject:
        example["content"] = inject_ghost(example["content"], to_inject[idx])

    return example
