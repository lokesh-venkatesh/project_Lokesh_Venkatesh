import random
import torch
import numpy as np
from config import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def set_seed():
    """
    Set the random seed for reproducibility. If a global SEED variable is 
    already defined, use it. Otherwise, set SEED to 42 and define it globally.
    """
    global SEED
    try:
        SEED
    except NameError:
        SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def set_available_device():
    """
    Determines the best available device for computation (GPU, MPS, or CPU) 
    and sets it to the global variable DEVICE. If a CUDA-enabled GPU is 
    available, it selects 'cuda'. If Metal Performance Shaders (MPS) are 
    available on macOS, it selects 'mps'. Otherwise, it defaults to 'cpu'.
    """
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE

def split_images(list_of_images):
    """
    Shuffle and split images into train, validation, and test sets.
    - images: List of image paths.
    - output: Dictionary containing lists of image paths for train, val, and test sets.
    """
    random.shuffle(list_of_images)
    n_total = len(list_of_images)
    n_train = int(n_total*split_ratios[0])
    n_val = int(n_total*split_ratios[1])
    return {"train": list_of_images[:n_train], 
            "val": list_of_images[n_train:n_train + n_val], 
            "test": list_of_images[n_train + n_val:]}

def count_parameters(pth_file):
    """
    Counts the total number of parameters in a PyTorch model saved in a .pth file.
    This function loads a PyTorch model from the specified .pth file and determines 
    whether the file contains a full model, a state dictionary, or a state dictionary 
    within another dictionary. It then calculates the total number of parameters 
    in the model.
    Args:
        pth_file (str): Path to the .pth file containing the PyTorch model or state dictionary.
    Returns:
        int: The total number of parameters in the model.
    """
    # Load the .pth file
    model = torch.load(pth_file, map_location=torch.device('cpu'))
    
    # Check if the model is a state_dict or a full model
    if isinstance(model, dict) and 'state_dict' in model:
        state_dict = model['state_dict']
    elif isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    
    # Count the total number of parameters
    total_params = sum(param.numel() for param in state_dict.values())
    return total_params

def calculate_accuracy(outputs, labels):
    """
    Calculate the accuracy of predictions compared to the true labels.
    Args:
        outputs (torch.Tensor): The raw model outputs (logits).
        labels (torch.Tensor): The ground truth labels.
    Returns:
        float: The accuracy as a percentage.
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def construct_confusion_matrix(outputs, labels):
    """
    Construct a confusion matrix for the given model outputs and true labels.
    Args:
        outputs (torch.Tensor): The raw model outputs (logits).
        labels (torch.Tensor): The ground truth labels.
    Returns:
        np.ndarray: The confusion matrix.
    """
    _, predicted = torch.max(outputs, 1)
    return confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

def plot_and_save_confusion_matrix(outputs, labels, filepath, class_names, caption="Confusion Matrix"):
    """
    Plot and save the confusion matrix to a local filepath with class names and a caption.
    Args:
        outputs (torch.Tensor): The raw model outputs (logits).
        labels (torch.Tensor): The ground truth labels.
        filepath (str): The path where the confusion matrix image will be saved.
        class_names (list): List of class names corresponding to the labels.
        caption (str): Caption for the confusion matrix plot.
    """
    conf_matrix = construct_confusion_matrix(outputs, labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(caption)
    #plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=0)
    plt.savefig(filepath, dpi=300)
    plt.close()

def calculate_precision(conf_matrix):
    """
    Calculate precision from the confusion matrix.
    Args:
        conf_matrix (np.ndarray): The confusion matrix.
    Returns:
        float: The precision.
    """
    true_positive = np.diag(conf_matrix)
    false_positive = np.sum(conf_matrix, axis=0) - true_positive
    precision = np.sum(true_positive / (true_positive + false_positive + 1e-10)) / len(true_positive)
    return precision

def calculate_recall(conf_matrix):
    """
    Calculate recall from the confusion matrix.
    Args:
        conf_matrix (np.ndarray): The confusion matrix.
    Returns:
        float: The recall.
    """
    true_positive = np.diag(conf_matrix)
    false_negative = np.sum(conf_matrix, axis=1) - true_positive
    recall = np.sum(true_positive / (true_positive + false_negative + 1e-10)) / len(true_positive)
    return recall

def calculate_f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall.
    Args:
        precision (float): The precision value.
        recall (float): The recall value.
    Returns:
        float: The F1 score.
    """
    return 2 * (precision * recall) / (precision + recall + 1e-10)

def plot_and_save_performance_curves(training_log, val_losses, val_accuracies, save_path):
        import matplotlib.pyplot as plt

        N = len(val_losses)
        epochs = [log[0] for log in training_log][0:N]
        train_losses = [log[1] for log in training_log][0:N]
        train_accuracies = [log[2] for log in training_log][0:N]

        plt.figure(figsize=(12, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Training Accuracy", marker="o")
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def count_parameters_in_pth(pth_path):
    """
    Count the number of parameters in a PyTorch .pth file.
    Args: pth_path (str): Path to the .pth file.
    Returns: int: Total number of parameters."""
    state_dict = torch.load(pth_path, map_location='cpu')
    # If it's a full checkpoint, get the 'state_dict' inside it
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    total_params = sum(param.numel() for param in state_dict.values())
    return total_params