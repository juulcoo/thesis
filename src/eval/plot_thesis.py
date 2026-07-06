import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

METHOD_LABELS = {
    "random": "Random",
    "high_loss": "High-loss",
    "high_logppl": "High-loss",
    "optimized": "Loss-drop",
    "learnability": "Loss-drop",
    "loss_drop": "Loss-drop",
}

METHOD_ORDER = {
    "random": 0,
    "high_loss": 1,
    "high_logppl": 1,
    "optimized": 2,
    "learnability": 2,
    "loss_drop": 2,
}


def pretty_method(method):
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def method_sort_key(method):
    return METHOD_ORDER.get(method, 99)

def load_finite_array(path):
    arr = np.load(path)
    arr = np.asarray(arr)
    return arr[np.isfinite(arr)]


def plot_roc_from_arrays(pos, neg, score_direction, label):
    if score_direction == "lower_member":
        y_score = np.concatenate([-pos, -neg])
    elif score_direction == "higher_member":
        y_score = np.concatenate([pos, neg])
    else:
        raise ValueError(f"Unknown score_direction: {score_direction}")

    y_true = np.concatenate([
        np.ones(len(pos)),
        np.zeros(len(neg)),
    ])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")


def plot_optimization_rocs(run_dirs, out_dir):
    run_dirs = sorted(
        run_dirs,
        key=lambda run_dir: method_sort_key(parse_run_name(run_dir.name)[0]),
    )

    plt.figure(figsize=(6.5, 5))

    for run_dir in run_dirs:
        tm_path = run_dir / "TM_ghost_logppl.npy"
        ntm_path = run_dir / "NTM_ghost_logppl.npy"

        if not tm_path.exists() or not ntm_path.exists():
            print(f"Skipping ghost ROC for {run_dir.name}: missing ghost arrays")
            continue

        tm = load_finite_array(tm_path)
        ntm = load_finite_array(ntm_path)

        plot_roc_from_arrays(
            pos=tm,
            neg=ntm,
            score_direction="lower_member",
            label=pretty_method(parse_run_name(run_dir.name)[0]),
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Ghost log-PPL ROC: TM vs NTM")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "optimization_ghost_logppl_roc.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")

def plot_mink_rocs(run_dirs, out_dir):
    run_dirs = sorted(
        run_dirs,
        key=lambda run_dir: method_sort_key(parse_run_name(run_dir.name)[0]),
    )

    comparisons = [
        (
            "TM vs NTM | Min-K 10%",
            "TM_mink10.npy",
            "NTM_mink10.npy",
            "optimization_mink_tm_vs_ntm_roc.png",
        ),
        (
            "T vs NT | Min-K 10%",
            "T_mink10.npy",
            "NT_mink10.npy",
            "optimization_mink_t_vs_nt_roc.png",
        ),
    ]

    for title, member_file, nonmember_file, filename in comparisons:
        plt.figure(figsize=(6.5, 5))

        for run_dir in run_dirs:
            member_path = run_dir / member_file
            nonmember_path = run_dir / nonmember_file

            if not member_path.exists() or not nonmember_path.exists():
                print(f"Skipping {title} ROC for {run_dir.name}: missing arrays")
                continue

            member = load_finite_array(member_path)
            nonmember = load_finite_array(nonmember_path)

            plot_roc_from_arrays(
                pos=member,
                neg=nonmember,
                score_direction="higher_member",
                label=pretty_method(parse_run_name(run_dir.name)[0]),
            )

        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved {out_path}")


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
    match = re.search(r"(.+)_mu(\d+)", name)

    if match is None:
        return name, None

    method = match.group(1)
    mu = int(match.group(2))

    return method, mu


def finite(x):
    x = np.asarray(x)
    return x[np.isfinite(x)]

def plot_optimization_grouped_bars(combined, out_dir, mu=1):
    metric_name = "TM vs NTM | ghost log-PPL"

    df = combined[
        (combined["name"] == metric_name)
        & (combined["mu"] == mu)
    ].copy()

    if df.empty:
        print("Skipping optimization grouped bars: no matching ghost log-PPL rows")
        return

    methods = sorted(df["method"].unique(), key=method_sort_key)

    metric_cols = [
        ("auc", "AUC"),
        ("tpr_at_1fpr", "TPR@1%FPR"),
        ("tpr_at_5fpr", "TPR@5%FPR"),
        ("tpr_at_10fpr", "TPR@10%FPR"),
    ]

    x = np.arange(len(metric_cols))
    width = 0.8 / max(1, len(methods))

    plt.figure(figsize=(8, 4.8))

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method]

        values = []
        for col, _ in metric_cols:
            if col in method_df.columns and not method_df.empty:
                values.append(float(method_df.iloc[0][col]))
            else:
                values.append(np.nan)

        offset = (i - (len(methods) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=pretty_method(method))

    plt.xticks(x, [label for _, label in metric_cols])
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title(r"Optimization comparison at $\mu=1$: TM vs NTM ghost log-PPL")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "optimization_ghost_logppl_grouped_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


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

    plot_optimization_grouped_bars(combined, out_dir, mu=1)
    plot_optimization_rocs(run_dirs, out_dir)
    plot_mink_rocs(run_dirs, out_dir)

if __name__ == "__main__":
    main()