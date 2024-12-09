import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_history(history):
    """
    Plots the training and validation accuracy/loss curves.

    Parameters:
    - history: A History object returned by the `model.fit()` function.
    """
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Parameters:
    - y_true: Ground truth (correct) target values.
    - y_pred: Estimated targets as returned by a classifier.
    - labels: Optional list of label names to display in the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics_comparison(baseline_metrics, ca_metrics, metric_names):
    """
    Plots a comparison of baseline and CA-based model metrics.

    Parameters:
    - baseline_metrics: List of metrics for the baseline model.
    - ca_metrics: List of metrics for the CA-based model.
    - metric_names: List of metric names.
    """
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, baseline_metrics, width, label='Baseline Model')
    plt.bar(x + width/2, ca_metrics, width, label='Cultural Algorithm Model')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Baseline and CA-Based Model Metrics')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.show()
