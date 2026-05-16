# thesis

All configurations (model, dataset, hyperparameters, etc.) can be set in config.yaml in the project root. The config is loaded in config.py to be imported

# TODO
- Training the model with smaller dataset
    - Set up differential privacy (feature toggle)
- MIA attacks
    - Last-k-words test
    - Perplexity test
- Watermarking

# Good questions
- Perplexity-based filtering is a common method for cleaning pre-training data. How might the proposed ghost sentences method be adapted to remain effective in the presence of perplexity-based data cleaning techniques commonly used in LLM training?
- With LLM-generated or rephrased training data becoming more popular, it would be interesting to investigate whether ghost sentences persist after LLM-based rewriting.