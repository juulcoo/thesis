from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def clean_pair(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)

    pos_scores = pos_scores[~np.isnan(pos_scores)]
    neg_scores = neg_scores[~np.isnan(neg_scores)]

    return pos_scores, neg_scores


def binary_metrics(pos_scores, neg_scores, name, higher_is_member=True, fpr_targets=(0.01, 0.05, 0.10)):
    """
    pos_scores: member / positive scores
    neg_scores: non-member / negative scores
    higher_is_member:
        True for Min-K logprob
        False for loss / perplexity
    """
    pos_scores, neg_scores = clean_pair(pos_scores, neg_scores)

    y_true = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores)),
    ])

    scores = np.concatenate([pos_scores, neg_scores])

    if not higher_is_member:
        scores = -scores

    auc_value = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    result = {
        "name": name,
        "auc": float(auc_value),
    }

    for target in fpr_targets:
        valid = np.where(fpr <= target)[0]

        if len(valid) == 0:
            result[f"tpr_at_{int(target * 100)}fpr"] = 0.0
            result[f"threshold_at_{int(target * 100)}fpr"] = None
            continue

        best_idx = valid[np.argmax(tpr[valid])]
        result[f"tpr_at_{int(target * 100)}fpr"] = float(tpr[best_idx])
        result[f"threshold_at_{int(target * 100)}fpr"] = float(thresholds[best_idx])

    return result


def auc(member_scores, nonmember_scores):
    return binary_metrics(
        member_scores,
        nonmember_scores,
        name="auc",
        higher_is_member=False,
    )["auc"]

def get_roc(member_scores, nonmember_scores):
    member_scores, nonmember_scores = clean_pair(member_scores, nonmember_scores)

    y_true = [1] * len(member_scores) + [0] * len(nonmember_scores)
    y_score = [-x for x in member_scores] + [-x for x in nonmember_scores]

    auc_value = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return fpr, tpr, auc_value

def print_metric_row(result):
    print(
        f"{result['name']} | "
        f"AUC={result['auc']:.4f} | "
        f"TPR@1%FPR={result['tpr_at_1fpr']:.4f} | "
        f"TPR@5%FPR={result['tpr_at_5fpr']:.4f} | "
        f"TPR@10%FPR={result['tpr_at_10fpr']:.4f}"
    )