import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
import shutil
from pathlib import Path
from config import *
from sklearn.model_selection import StratifiedKFold
from utils import set_available_device, set_seed, split_images

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """Initialize the dataset with the root directory and optional transformations."""
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(label_idx)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieve an image and its label by index."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("L")  # convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

def inferLoader(batch_dir=predict_data_dir, task="predict"):
    if task == "predict":
        transform = transforms.Compose([
            transforms.Resize((image_resize_x, image_resize_y)),
            transforms.ToTensor(),
            transforms.Normalize([image_normalise], [image_normalise])
        ])
        batch_size = prediction_batch_size
        shuffle_toggle = False
        use_sampler = False
    elif task == "train":
        transform = transforms.Compose([
            transforms.Resize((image_resize_x, image_resize_y)),
            transforms.ToTensor(),
            transforms.Normalize([image_normalise], [image_normalise]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(image_random_rot)
        ])
        batch_size = training_batch_size
        shuffle_toggle = False  # Important: Set False if using a sampler
        use_sampler = True

    batch_dataset = ChestXRayDataset(root_dir=batch_dir, transform=transform)

    if task == "train" and use_sampler:
        # Compute class weights
        labels = [label for _, label in batch_dataset]
        label_counts = Counter(labels)
        total_samples = len(batch_dataset)
        class_weights = {cls: total_samples / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        batch_loader = DataLoader(batch_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=0, pin_memory=True)
    else:
        batch_loader = DataLoader(batch_dataset, batch_size=batch_size,
                                  shuffle=shuffle_toggle, num_workers=0, pin_memory=True)

    return batch_loader


def get_all_dataloaders(split_dataset_path=data_split_dir):
    """
    Create data loaders for training, validation, and test sets.
    Data loaders are basically lists of images and labels that are used to feed the model during training and evaluation.
    - data_dir: Directory containing the dataset.
    - batch_size: Number of samples per batch.
    - train_loader: DataLoader for training set.
    - val_loader: DataLoader for validation set.
    - test_loader: DataLoader for test set.
    - class_names: List of class names.
    """
    train_dir = os.path.join(split_dataset_path, "train")
    val_dir = os.path.join(split_dataset_path, "val")
    test_dir = os.path.join(split_dataset_path, "test")

    train_loader = inferLoader(batch_dir=train_dir, task="train")
    val_loader = inferLoader(batch_dir=val_dir, task="predict")
    test_loader = inferLoader(batch_dir=test_dir, task="predict")
    
    class_names = sorted(os.listdir(train_dir))

    return train_loader, val_loader, test_loader, class_names

def split_the_dataset():
    os.makedirs(data_split_dir, exist_ok=True)
    os.makedirs(model_params_dir, exist_ok=True)
    
    for category in ["train", "val", "test"]:
        for class_name in os.listdir(raw_dataset_dir):
            Path(f"{data_split_dir}/{category}/{class_name}").mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in os.listdir(raw_dataset_dir):
        class_path = Path(raw_dataset_dir) / class_name
        images_list = sorted(class_path.glob("*"))
        
        split_images_list = split_images(images_list)
        for category, files in split_images_list.items():
            for file in files:
                shutil.copy(file, f"{data_split_dir}/{category}/{class_name}/{file.name}")
        print(f"✅ {class_name} dataset split completed.")

    print("✅ Dataset split completed.")

    # Uncomment the following line to create dataloaders
    train_loader, val_loader, test_loader, class_names = get_all_dataloaders(data_split_dir)
    print(f"🔍 Number of classes: {len(class_names)}")
    print(f"🔍 Number of training samples: {len(train_loader.dataset)}")
    print(f"🔍 Number of validation samples: {len(val_loader.dataset)}")
    print(f"🔍 Number of testing samples: {len(test_loader.dataset)}")
    print(f"🔍 Class names: {class_names}")

def get_kfold_dataloaders(dataset_dir=raw_dataset_dir, n_splits=5, current_fold=0):
    """
    Generate train and validation dataloaders for a specific fold using Stratified K-Fold Cross-Validation.
    """
    transform_train = transforms.Compose([
        transforms.Resize((image_resize_x, image_resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(image_random_rot),
        transforms.ToTensor(),
        transforms.Normalize([image_normalise], [image_normalise])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((image_resize_x, image_resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([image_normalise], [image_normalise])
    ])

    # Load entire dataset
    full_dataset = ChestXRayDataset(root_dir=dataset_dir, transform=None)

    X = full_dataset.image_paths
    y = full_dataset.labels

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))
    train_idx, val_idx = splits[current_fold]

    # Build subset datasets
    train_samples = [(X[i], y[i]) for i in train_idx]
    val_samples = [(X[i], y[i]) for i in val_idx]

    # Custom Dataset class from sample lists
    class SubsetChestXRayDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("L")
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = SubsetChestXRayDataset(train_samples, transform=transform_train)
    val_dataset = SubsetChestXRayDataset(val_samples, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    class_names = sorted(os.listdir(dataset_dir))

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    set_seed()
    DEVICE = set_available_device()
    split_the_dataset()