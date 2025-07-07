import os
import matplotlib.pyplot as plt
import numpy as np
import random

def save_plot(fig, path, tight_layout=True):
    """
    Save a matplotlib figure to the specified path.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save
        path (str): File path to save the image
        tight_layout (bool): Whether to use tight layout before saving
    """
    if tight_layout:
        fig.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig.savefig(path)
    print(f"✅ Plot saved to: {path}")


def set_seed(seed=42):
    """
    Set random seed across numpy, random, and torch (if available).

    Parameters:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("🔒 PyTorch seed set.")
    except ImportError:
        print("ℹ️ PyTorch not installed — skipping torch seed.")


def ensure_dir(directory):
    """
    Ensure a directory exists; create it if it doesn't.

    Parameters:
        directory (str): Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")
    else:
        print(f"📂 Directory exists: {directory}")
