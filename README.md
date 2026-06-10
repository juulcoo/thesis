# thesis

## Setup

1. Baseline Replication
    - Instruction tuning setup with ghost sentences. Check insertion and repitition. Perplexity AUC and recall etc.
    - To compare with the baseline paper.

2. Preprocessing robustness
    - Train models on the data after realistic cleaning steps.
    - Deduplication, Differential Privacy, LLM rewriting, random sentence deletion, etc.
    - Show what types of text identifiers survive realistic preprocessing pipelines and which fail. (!!)

3. Compare different identifiers
    - Ghost sentences
    - Rare token ghosts
    - Unicode ghosts (From 2024 paper)
    - ...

Best memorization + Lowest perplexity

```
Do Ghost Sentences Survive the Web? Robustness of Natural-Language Data Watermarks for LLM Training Detection
```


## Data construction

1. Main dataset
    - Webis tldr17
    - Loaded in load_base.py
2. 148k subset
    - 148k examples of main dataset
    - Only users with 10-200 amount of docs
    - Loaded in load_subset.py
3. 
4. Ghost sentence library
    - We create 500 ghost sentences of length 10
    - Created in ghosts/create_ghosts.py
    - Ghosts are loaded in ghosts/load_ghosts.py

### Good questions
- Perplexity-based filtering is a common method for cleaning pre-training data. How might the proposed ghost sentences method be adapted to remain effective in the presence of perplexity-based data cleaning techniques commonly used in LLM training?
- With LLM-generated or rephrased training data becoming more popular, it would be interesting to investigate whether ghost sentences persist after LLM-based rewriting.