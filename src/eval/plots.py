from .metrics import get_roc
import matplotlib.pyplot as plt

def print_roc_results(T_scores, TM_scores, NT_scores, NTM_scores):
    _, _, auc_TM = get_roc(TM_scores, NT_scores)
    _, _, auc_T = get_roc(T_scores, NT_scores)
    _, _, auc_NTM = get_roc(NTM_scores, NT_scores)

    print(f"AUC for TM vs NT: {auc_TM:.4f}")
    print(f"AUC for T vs NT: {auc_T:.4f}")
    print(f"AUC for TM vs NTM: {auc_NTM:.4f}")

    print(f"T mean loss:  {T_scores.mean():.4f}")
    print(f"TM mean loss: {TM_scores.mean():.4f}")
    print(f"NT mean loss: {NT_scores.mean():.4f}")
    print(f"NTM mean loss: {NTM_scores.mean():.4f}")

def plot_rocs(T_scores, TM_scores, NT_scores, NTM_scores, save_path="roc_curves.png"):
    fpr_t, tpr_t, auc_t = get_roc(T_scores, NT_scores)
    fpr_tm, tpr_tm, auc_tm = get_roc(TM_scores, NT_scores)
    fpr_ntm, tpr_ntm, auc_ntm = get_roc(NTM_scores, NT_scores)

    plt.figure()
    plt.plot(fpr_t, tpr_t, label=f"T vs NT AUC = {auc_t:.3f}")
    plt.plot(fpr_tm, tpr_tm, label=f"TM vs NT AUC = {auc_tm:.3f}")
    plt.plot(fpr_ntm, tpr_ntm, label=f"TM vs NTM AUC = {auc_ntm:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Membership Inference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()