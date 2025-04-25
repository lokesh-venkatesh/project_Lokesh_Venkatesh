# interface.py contains the essential classes, functions and variable names
# used in this pipeline.

# replace MyCustomModel with the name of your model
from model import ChestXRayModel as TheModel

# change my_descriptively_named_train_function to
# the function inside train.py that runs the training loop.
from train import train_model as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import return_preds_but_only_as_labelslist as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import ChestXRayDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import inferLoader as the_dataloader

# change batchsize, epochs to whatever your names are for these
# variables inside the config.py file
from config import training_batch_size as the_batch_size
from config import number_of_epochs as total_epochs