import os
from torchvision import transforms

# Optional Early Stop
early_stop_patience = 8
early_stop_counter = 0

# Train, test and val split
train_split = 0.8
val_split = 0.1
test_split = 0.1
split_ratios = [train_split, val_split, test_split]

# model params
image_resize_x = 224
image_resize_y = 224
image_random_rot = 2
image_normalise = 0.5
n_input_channels = 1
preds_transform = transforms.Compose([transforms.Resize((image_resize_x, image_resize_y)), transforms.ToTensor(), transforms.Normalize([image_normalise], [image_normalise])])

# training hyperparams
number_of_epochs = 3
training_batch_size = 32
prediction_batch_size = 1024
learning_rate = 1e-4

# random seed and device
SEED = 42

# directories and filepaths
raw_dataset_dir = "Chest_X_Ray_Dataset"
data_split_dir = "split"
model_params_dir = "checkpoints"
predict_data_dir = "data"
weights_filepath = f"{model_params_dir}/final_weights.pth"
logs_dir = "logs"
pred_images = [os.path.join(predict_data_dir, filename) for filename in os.listdir(predict_data_dir)]
class_names = ['Covid19', 'Pneumonia', 'Normal', 'Tuberculosis']