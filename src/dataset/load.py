from dataset.load_base import load_ds
from dataset.load_subset import load_subset
from dataset.load_splits import load_splits

def load():
    base = load_ds()                                    # Has normal webis-tldr17 columns and content
    subset = load_subset(base)                          # Cut down to 148k examples; Columns = {"content", "author", "id"}

    T, TM, NT, training_set = load_splits(subset)       # Loads 10k examples for each split; Columns = {"content", "author", "id", "has_ghost", "ghost", "original_content"}}

    verify(T, TM, NT)
    
    return T, TM, NT, training_set
    
def verify(T, TM, NT):
    print("T:", T)
    print("TM:", TM)
    print("NT:", NT)
