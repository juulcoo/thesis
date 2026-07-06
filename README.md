# Thesis project

This repository trains and evaluates language models with ghost-sentence watermarking experiments. The main workflow is:

1. Generate or select ghost sentences.
2. Train a model on the prepared dataset.
3. Run evaluation metrics and MIA analysis.
4. Plot the results for comparison.

The project is configured through [config.yaml](config.yaml) and driven by the shell scripts in [scripts](scripts).

## 1. Environment setup

### Create a virtual environment

```bash
cd /path/to/thesis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

If you use Hugging Face models or datasets, it is also helpful to set cache directories:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"
```

You can add these to your shell profile if you want them to persist.

## 2. Configuration

The main settings live in [config.yaml](config.yaml). Before running anything, review at least:

- model name and output directory
- dataset settings
- ghost generation settings
- optimization hyperparameters

The training and optimization scripts also assume a repository path of `~/thesis` and a virtual environment at `~/thesis/.venv`. If your checkout is elsewhere, update those paths in the shell scripts under [scripts](scripts).

## 3. How the scripts work

The scripts in [scripts](scripts) are the main entry points.

### train.sh

Runs the standard training pipeline:

```bash
python -m training.train
python -m eval.mia
```

It:

- clears generated data
- activates the virtual environment
- sets `PYTHONPATH=src`
- trains the model
- runs membership inference evaluation and writes logs

Use this when you want to train the base model and evaluate it.

### optimize.sh

Runs the learnability-based ghost optimization workflow:

```bash
python -m data.optimize_learnability
python -m training.train
python -m eval.mia
```

This script first selects optimized ghost sentences and then trains and evaluates the model using those ghosts.

### optimize_loss.sh

Runs the high-loss ghost selection workflow:

```bash
python -m data.optimize_high_loss
python -m training.train
python -m eval.mia
```

This is similar to the learnability script, but it selects ghosts based on high-loss criteria.

### plot_runs.sh

Generates plots from completed experiment runs.

It expects one or more run directories containing `metrics.csv` files. If no arguments are provided, it uses the folders under `results/named/*`.

Example:

```bash
bash scripts/plot_runs.sh results/named/random_vs_learnability_mu1_n1000
```

## 4. Typical run order

1. Activate the environment.
2. Review and edit [config.yaml](config.yaml).
3. Choose one of the optimization scripts or run the training script directly.
4. Run the evaluation and plotting steps.

### Example workflow

```bash
source .venv/bin/activate
export PYTHONPATH=src

# Option A: train directly
bash scripts/train.sh

# Option B: generate optimized ghosts and train
bash scripts/optimize.sh

# Option C: generate high-loss ghosts and train
bash scripts/optimize_loss.sh

# Plot a completed run
bash scripts/plot_runs.sh results/named/<your_run_dir>
```

If you are not using Slurm, you can still run the same Python commands manually from the repository root.

## 5. Project structure

- [src](src): core training, data generation, and evaluation code
- [data](data): wordlists, generated ghosts, and dataset artifacts
- [results](results): experiment outputs and plots
- [scripts](scripts): shell entry points for training, optimization, and plotting

## 6. Dataset and ghost generation notes

The project uses a main dataset based on Webis TLDR-17 and generates ghost sentences from a wordlist. The optimization scripts create ghost candidates and save them to [data/generated](data/generated), while the training step reads those generated artifacts before training.

If you want to understand the experiment goals, the project revolves around comparing:

- random ghosts
- high-loss ghosts
- learnability-optimized ghosts

and measuring how well they survive or influence memorization and evaluation metrics.
