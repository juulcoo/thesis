import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(run_dir):
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {run_dir}")

    df = pd.read_csv(path)
    df["run_dir"] = str(run_dir)
    df["run_name"] = run_dir.name

    method, mu = parse_run_name(run_dir.name)
    df["method"] = method
    df["mu"] = mu

    return df


def parse_run_name(name):
    """
    Expected names like:
      random_mu1_1000
      high_logppl_mu3_3333
      learnability_mu1_1000

    If no pattern is found, method=name and mu=None.
    """

    match = re.search(r"(.+)_mu(\d+)", name)

    if match is None:
        return name, None

    method = match.group(1)
    mu = int(match.group(2))

    return method, mu


def finite(x):
    x = np.asarray(x)
    return x[np.isfinite(x)]


def plot_ghost_logppl_distribution(run_dir, out_dir):
    tm_path = run_dir / "TM_ghost_logppl.npy"
    ntm_path = run_dir / "NTM_ghost_logppl.npy"

    if not tm_path.exists() or not ntm_path.exists():
        print(f"Skipping distribution plot for {run_dir.name}: missing ghost log-PPL arrays")
        return

    tm = finite(np.load(tm_path))
    ntm = finite(np.load(ntm_path))

    plt.figure(figsize=(7, 4.5))

    bins = 50
    plt.hist(ntm, bins=bins, alpha=0.6, density=True, label="NTM: untrained marked ghosts")
    plt.hist(tm, bins=bins, alpha=0.6, density=True, label="TM: trained marked ghosts")

    plt.xlabel("Ghost log-perplexity")
    plt.ylabel("Density")
    plt.title(f"Ghost log-PPL distribution: {run_dir.name}")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{run_dir.name}_ghost_logppl_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def plot_metric_over_mu(combined, metric_name, value_col, out_dir):
    df = combined[combined["name"] == metric_name].copy()
    df = df.dropna(subset=["mu", value_col])

    if df.empty:
        print(f"Skipping {value_col} over μ: no data for {metric_name}")
        return

    plt.figure(figsize=(7, 4.5))

    for method, group in df.groupby("method"):
        group = group.sort_values("mu")
        plt.plot(group["mu"], group[value_col], marker="o", label=method)

    plt.xlabel("Repetition μ")
    plt.ylabel(value_col)
    plt.title(f"{metric_name}: {value_col} over μ")
    plt.legend()
    plt.tight_layout()

    safe_metric = metric_name.replace(" ", "_").replace("|", "").replace("/", "_")
    out_path = out_dir / f"{safe_metric}_{value_col}_over_mu.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def plot_main_bar(combined, out_dir):
    df = combined[combined["name"] == "TM vs NTM | ghost log-PPL"].copy()

    if df.empty:
        print("Skipping main bar plot: no ghost log-PPL rows")
        return

    df = df.sort_values(["mu", "method"])

    labels = [
        f"{row.method}\nμ={int(row.mu) if not pd.isna(row.mu) else '?'}"
        for row in df.itertuples()
    ]

    x = np.arange(len(df))

    plt.figure(figsize=(max(7, len(df) * 0.8), 4.5))
    plt.bar(x, df["auc"])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title("Ghost log-PPL AUC by method")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    out_path = out_dir / "ghost_logppl_auc_by_method.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def plot_learnability_diagnostic(run_dir, out_dir):
    score_path = run_dir / "learnability_scores.csv"

    if not score_path.exists():
        print(f"Skipping learnability diagnostic for {run_dir.name}: no learnability_scores.csv")
        return

    df = pd.read_csv(score_path)

    required = {"logppl", "learnability_drop"}

    if not required.issubset(df.columns):
        print(f"Skipping learnability diagnostic for {run_dir.name}: missing required columns")
        return

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["logppl", "learnability_drop"])

    if df.empty:
        print(f"Skipping learnability diagnostic for {run_dir.name}: no finite rows")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(df["logppl"], df["learnability_drop"], s=8, alpha=0.4)

    plt.xlabel("Base ghost log-perplexity")
    plt.ylabel("One-step log-PPL drop")
    plt.title(f"Learnability diagnostic: {run_dir.name}")
    plt.tight_layout()

    out_path = out_dir / f"{run_dir.name}_learnability_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def make_combined_table(run_dirs, out_dir):
    dfs = []

    for run_dir in run_dirs:
        try:
            dfs.append(load_metrics(run_dir))
        except FileNotFoundError as e:
            print(e)

    if not dfs:
        raise ValueError("No metrics.csv files found.")

    combined = pd.concat(dfs, ignore_index=True)

    out_path = out_dir / "combined_results.csv"
    combined.to_csv(out_path, index=False)

    print(f"Saved {out_path}")

    main_rows = combined[
        combined["name"].isin(
            [
                "TM vs NTM | ghost log-PPL",
                "TM vs NTM | Min-K 10%",
                "T vs NT | Min-K 10%",
            ]
        )
    ].copy()

    cols = [
        "method",
        "mu",
        "name",
        "auc",
        "tpr_at_1fpr",
        "tpr_at_5fpr",
        "tpr_at_10fpr",
        "member_median",
        "nonmember_median",
        "directional_median_gap",
    ]

    existing_cols = [col for col in cols if col in main_rows.columns]
    main_rows = main_rows[existing_cols]

    table_path = out_dir / "main_thesis_table.csv"
    main_rows.to_csv(table_path, index=False)

    print(f"Saved {table_path}")

    return combined


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Result directories, e.g. results/named/random_mu1_1000 results/named/learnability_mu1_1000",
    )

    parser.add_argument(
        "--out-dir",
        default="results/figures",
        help="Where to save figures and combined tables.",
    )

    args = parser.parse_args()

    run_dirs = [Path(path) for path in args.run_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = make_combined_table(run_dirs, out_dir)

    for run_dir in run_dirs:
        plot_ghost_logppl_distribution(run_dir, out_dir)
        plot_learnability_diagnostic(run_dir, out_dir)

    plot_main_bar(combined, out_dir)

    plot_metric_over_mu(
        combined,
        metric_name="TM vs NTM | ghost log-PPL",
        value_col="auc",
        out_dir=out_dir,
    )

    plot_metric_over_mu(
        combined,
        metric_name="TM vs NTM | ghost log-PPL",
        value_col="tpr_at_1fpr",
        out_dir=out_dir,
    )

    plot_metric_over_mu(
        combined,
        metric_name="TM vs NTM | ghost log-PPL",
        value_col="directional_median_gap",
        out_dir=out_dir,
    )

if __name__ == "__main__":
    main()