import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def evaluate_and_plot(y_true, y_pred, fscore_beta=1, threshold=0.5, plot_prec_recall_curve=True):
    """Plots a Confusion Matrix and optionally a Precision-Recall Curve. Prints Accuracy, Fscore, Recall and Precision.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        fscore_beta (int, optional): _description_. Defaults to 1.
        threshold (float, optional): _description_. Defaults to 0.5.
        plot_prec_recall_curve (bool, optional): _description_. Defaults to True.
    """
    # obtain and plot confusion_matrix
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    y_true = [int(k) for k in y_true]
    disp = ConfusionMatrixDisplay.from_predictions(y_true, [(k > threshold) * 1 for k in y_pred], labels=[0, 1], ax=ax)
    ax.set_xticklabels(["Not AI", "AI"])
    ax.set_yticklabels(["Not AI", "AI"])
    plt.show()

    if plot_prec_recall_curve:
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=ax)
        plt.show()

    # Print Fscore, Recall and Precision, formulas checked from https://en.wikipedia.org/wiki/F-score
    cm = disp.confusion_matrix
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fscore = (1 + fscore_beta ** 2) * (precision * recall) / ((fscore_beta ** 2 * precision) + recall)
    print(
        f"Accuracy: {accuracy:.2f}\tF{fscore_beta:.1f}-Score: {fscore:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}"
    )
