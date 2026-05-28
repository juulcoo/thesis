# thesis

## Data construction

1. Main dataset
    - Webis tldr17
    - Loaded in load_base.py
2. 148k subset
    - 148k examples of main dataset
    - Only users with 10-200 amount of docs
    - Loaded in load_subset.py
3. Task dataset
    - Subset split into half
    - First half is context
    - Second half is supposed to be learnt
    - Second half will contain the ghost sentence
4. Ghost sentence library
    - We create 500 ghost sentences of length 10
    - Created in ghosts/create_ghosts.py
    - Ghosts are loaded in ghosts/load_ghosts.py

From the paper:
```
Dataset Webis-TLDR-17 [Völske et al., 2017] contains 3.7M examples with word lengths under 4096.
Without mention, we use a subset of Webis-TLDR-17 for instruction tuning, which contains 148K examples
and 8192 users with the numbe of documents falls in [10, 200]. We term this subset as Webis-148K for
convenient. For instruction tuning on Webis-148K, LLMs are required to finish a continue writing task using
the instruction "Continue writing the given content". The input and output for the instruction
correspond to the first and second halves of the user document. For continuing pre-training, we also utilize
the LaMini-Instruction [Wu et al., 2023] and OpenOrca [Longpre et al., 2023, Mukherjee et al., 2023, Lian
et al., 2023] datasets, which contain 2.6M and 3.5M examples, respectively. Plus the Webis-TLDR-17 dataset,
the number of pre-training examples is 9.8M. All data are shuffled during training
```

All configurations (model, dataset, hyperparameters, etc.) can be set in config.yaml in the project root. The config is loaded in config.py to be imported

### Good questions
- Perplexity-based filtering is a common method for cleaning pre-training data. How might the proposed ghost sentences method be adapted to remain effective in the presence of perplexity-based data cleaning techniques commonly used in LLM training?
- With LLM-generated or rephrased training data becoming more popular, it would be interesting to investigate whether ghost sentences persist after LLM-based rewriting.