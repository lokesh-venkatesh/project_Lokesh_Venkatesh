import os
import torch
import csv
from PIL import Image
from dataset import inferLoader
from model import ChestXRayModel
from config import *
from utils import set_available_device, set_seed

# Setup
DEVICE = set_available_device()
set_seed()
os.makedirs(logs_dir, exist_ok=True)

# ---------- Averaging Model Weights ----------
def return_model_weights():
    model = ChestXRayModel(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(weights_filepath, map_location=DEVICE))
    model.eval()
    return model

# Load final averaged model
model = return_model_weights()

# -------- Batch Prediction --------
def batch_predict_images(image_paths=pred_images):
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset

    class InMemoryDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            image = Image.open(path).convert("L")
            if self.transform:
                image = self.transform(image)
            return image, path

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = InMemoryDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = {}
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for path, pred in zip(paths, preds):
                predictions[path] = class_names[pred.item()]
                print(f"Processed {os.path.basename(path)}: Predicted as {class_names[pred.item()]}")

    return predictions

# -------- Evaluation Functions --------
def organise_preds_into_outputs_and_labels(image_files_directory_list=pred_images):
    predictions = batch_predict_images(image_files_directory_list)
    predicted_labels = [class_names.index(predictions[path]) for path in image_files_directory_list]
    true_labels = [class_names.index(os.path.basename(path).split("_")[0]) for path in image_files_directory_list]
    predicted_labels = torch.tensor(predicted_labels, dtype=torch.long)
    true_labels = torch.tensor(true_labels, dtype=torch.long)
    return predicted_labels, true_labels

def return_preds_but_only_as_labelslist(image_files_directory=pred_images):
    predictions = batch_predict_images(image_files_directory)
    predicted_labels = [predictions[path] for path in image_files_directory]
    return predicted_labels

# -------- Script Entry --------
if __name__ == "__main__":
    pred_images = [os.path.join("data", filename) for filename in os.listdir("data") if filename.lower().endswith((".png", ".jpg", ".jpeg"))]
    predicted_labels, true_labels = organise_preds_into_outputs_and_labels(pred_images)

    #print("🔍 Predicted Labels:", [class_names[i] for i in predicted_labels.tolist()])
    #print("🔍 True Labels:", [class_names[i] for i in true_labels.tolist()])

    output_csv = f"{logs_dir}/predictions_and_labels.csv"
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Predicted Label", "True Label"])
        for image_path, pred, true in zip(pred_images, predicted_labels.tolist(), true_labels.tolist()):
            writer.writerow([image_path, class_names[pred], class_names[true]])

    print(f"✅ Predictions and labels saved to {output_csv}")