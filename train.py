import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from model import ChestXRayModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import ChestXRayDataset, get_all_dataloaders, inferLoader
from config import *
from utils import *
import csv

DEVICE = set_available_device()
set_seed()

train_loader, val_loader, test_loader, class_names = get_all_dataloaders()

CNN_model = ChestXRayModel(num_classes=len(class_names))
CNN_model.to(DEVICE)

if os.path.exists(weights_filepath):
    print(f"🔄 Loading pre-trained weights from {weights_filepath}")
    CNN_model.load_state_dict(torch.load(weights_filepath, map_location=DEVICE))
else:
    print("🆕 No pre-trained weights found, starting from scratch.")

loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.AdamW(CNN_model.parameters(), lr=learning_rate)


# Add scheduler: reduce LR if validation loss doesn't improve for 3 epochs
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def train_model(model=CNN_model, num_epochs=number_of_epochs, train_loader=train_loader, loss_fn=loss_fn, optim=optimizer):
    best_val_loss = float("inf")
    
    training_log = []
    val_losses = []
    val_accuracies = []

    # Ensure model parameters are on the correct device
    for param in model.parameters():
        param = param.to(DEVICE)

    for epoch in range(num_epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss, total_correct = 0, 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optim.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()

            total_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / len(train_loader.dataset)
        training_log.append((epoch + 1, avg_loss, avg_acc))
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_outputs, val_labels = 0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += loss_fn(outputs, labels).item()
                val_outputs.append(outputs)
                val_labels.append(labels)

        val_loss /= len(val_loader)
        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels)

        # Calculate metrics
        conf_matrix = construct_confusion_matrix(val_outputs, val_labels)
        precision = calculate_precision(conf_matrix)
        recall = calculate_recall(conf_matrix)
        f1_score = calculate_f1_score(precision, recall)
        accuracy = calculate_accuracy(val_outputs, val_labels)

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        # Save best model
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_filepath)
            print("💾 Saved new best model!")
            early_stop_counter = 0  # Reset counter if we improved
        else:
            early_stop_counter += 1
            print(f"⏸️ Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered!")
                break
            
        # Append validation metrics after each epoch
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

    # Save confusion matrix
    plot_and_save_confusion_matrix(val_outputs, val_labels, f"{logs_dir}/confusion_matrix.png", class_names)

    # Plot and save performance curves
    performance_curves_filepath = f"{logs_dir}/performance_curves.png"
    plot_and_save_performance_curves(training_log, val_losses, val_accuracies, performance_curves_filepath)
    print(len(training_log), training_log)
    print(len(val_losses), val_losses)
    print(len(val_accuracies), val_accuracies)

    # Save epoch-wise metrics to a CSV file
    epoch_metrics_filepath = f"{logs_dir}/epoch_metrics.csv"
    with open(epoch_metrics_filepath, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epoch", "Loss", "Accuracy"])
        for log in training_log:
            csvwriter.writerow([log[0], f"{log[1]:.4f}", f"{log[2]:.4f}"])

    # Save final training metrics to a separate CSV file
    final_metrics_filepath = f"{logs_dir}/final_metrics.csv"
    with open(final_metrics_filepath, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Metric", "Value"])
        csvwriter.writerow(["Precision", f"{precision:.4f}"])
        csvwriter.writerow(["Recall", f"{recall:.4f}"])
        csvwriter.writerow(["F1 Score", f"{f1_score:.4f}"])
        csvwriter.writerow(["Accuracy", f"{accuracy:.4f}"])

if __name__ == "__main__":
    print(f"✅ Using device: {DEVICE}")

    os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    train_model()