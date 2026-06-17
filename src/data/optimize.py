"""
You want a ghost sentence with high ghost loss before training and low ghost loss after training
We dont just minimize ghost losses on the base model, then we get sentences that are easy to predict. 

We can pre select a ghost sentence or optimize one with very high loss before training to get a hard to predict ghost sentence
We can optimize this sentence to have low loss after training, in other words 'High Learnability' 

# Optimize for low loss after training

If the ghost creates a large gradient, one training step should reduce its loss more strongly (we want strong reduction in loss during training in contrast to high init. loss).

We want: High initial loss + big gradient norm

J(g) = (ghost loss) l(g|c) + a log (||delta l(g|c)|| + e)

1.
    We can make a list of ghost sentences and for each get their initial loss, gradient norm and then score them
    We pick the best performing ones and insert those.
    We then compare this against random ghosts

2. 
    We can also go per position and take the scores of candidate words
    Calculate the change in loss when replacing current word with candidate word
    We start with a random word, compute the gradient of the ghost with respect to ghost embeddings
    For each position, score possible replacements
    Pick the best one and then repeat
"""

def optimize_initial_loss():
    pass

def optimize_gradient_norm():
    pass