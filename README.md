# thesis

## Setup

- Hypothesis:   
    - If the model never saw ghost sentence during training, it should be trash
    - If the model did see it (more often), the probabilities become high -> lower perplexity (not for other models, filtering)

Component	        Your baseline
Task	            Membership inference via perplexity
Identifier	        Ghost sentence
Membership signal	Reduced perplexity
Null hypothesis	    Model never saw identifier
Alternative	        Identifier repeated in training
Evaluation	        PPL threshold / ROC-AUC
Main variable	    repetition count μ

# Data construction

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