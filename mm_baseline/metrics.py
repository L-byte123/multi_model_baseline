import numpy as np


def classification_metrics(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    tp = matrix.diagonal().astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros_like(tp), where=matrix.sum(0) != 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros_like(tp), where=matrix.sum(1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=precision+recall != 0)
    return {"accuracy": float(tp.sum()/max(matrix.sum(), 1)), "macro_f1": float(f1.mean()),
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(), "confusion_matrix": matrix.tolist(), "samples": int(matrix.sum())}
