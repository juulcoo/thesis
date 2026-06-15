from sklearn.metrics import roc_curve, roc_auc_score

def get_roc(member_scores, nonmember_scores):
    y_true = [1] * len(member_scores) + [0] * len(nonmember_scores)
    y_score = [-x for x in member_scores] + [-x for x in nonmember_scores]

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return fpr, tpr, auc