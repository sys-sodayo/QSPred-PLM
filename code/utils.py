import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc
)


def evaluate_model_performance(true_labels, predictions):
    """Evaluate model performance on classification metrics and plot confusion matrix."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = recall_score(true_labels, predictions, pos_label=0)
    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)

    print(f"Acc: {accuracy:.4f}, "
          f"MCC: {mcc:.4f}, "
          f"Pr: {precision:.4f}, "
          f"Sn: {recall:.4f}, "
          f"Sp: {specificity:.4f}, "
          f"F1-score: {f1:.4f}, ", end="")

    plot_confusion_matrix(true_labels, predictions)


def plot_confusion_matrix(true_labels, predictions):
    """Plot the confusion matrix for given true and predicted labels."""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display counts in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

def roc(test_y, pr_list):
    """Plot ROC curve and calculate AUC score."""
    fpr, tpr, thresholds = roc_curve(test_y, pr_list)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xticks([(i / 10) for i in range(11)])
    plt.yticks([(i / 10) for i in range(11)])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print(f"AUC: {roc_auc:.4f} ")







