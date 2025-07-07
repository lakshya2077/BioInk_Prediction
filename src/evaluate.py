from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Print standard classification metrics.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.4f}\n")
    print("📋 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path=None):
    """
    Plot and optionally save a confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: Optional list of class labels (e.g., [0, 1])
    - title: Title of the plot
    - save_path: If provided, saves the plot instead of showing
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Auto-detect labels if not provided
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Plotting
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Confusion matrix saved to: {save_path}")
    else:
        plt.show()
