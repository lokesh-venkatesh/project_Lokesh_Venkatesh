# Machine Learning Pipeline to classify Images of Chest X-Ray Scans into categories based on the underlying medical condition

### About:
This repository was created as part of the final project for the "Image and Video Processing using Deep Learning" course offered by Dr. Chaitanya Guttikar in the Jan, 2025 semester at IISER Pune, by Lokesh Venkatesh (20221150) from the 3rd year batch of BS-MS Undergraduates. 

---

## Problem Introduction:
Lung disease diagnostics have improved in accuracy in recent years thanks to advances in medical imaging and classification. Chest X-ray (CXR) images are a widely used diagnostic tool, but interpreting them requires heavy expertise and is prone to human error. Automating the classification of chest X-rays can assist radiologists and improve early detection, diagnosis and treatment. We will try to develop a Convolutional Neural Network (CNN) model that can with reasonably high accuracy classify chest X-ray images into one of four labelled categories in the chosen dataset. This pipeline deals with a supervised classification problem using Convolutional Neural Network models. 

---

### Problem formulated as an input-output statement:
1. **Model Input:** A (unformatted) grayscale chest X-ray image given in .jpg format 
2. **Model Output:** A categorical classification label (Normal, Pneumonia, Tuberculosis, or Covid) corresponding to the actual diagnosed medical condition

---

## Choice of model:
In the original project proposal, I had originally wanted to use ResNet-50, which usually turns out to be the best balance of computational performance and accuracy. However, after training that model on the dataset, it did not give as promising of an accuracy. Ideally, since the pipeline is meant in medical image classification and diagnosis, the pipeline must ideally be close to 100% accuracy, if not perfect. Therefore, I decided to try the DenseNet-121 model, which has several deep convolutional layers along with a classifier attached at the end of the pipeline. This gave an overall accuracy of 96% and an F1 score of 95%, which turned out classify most of the scans with an underlying condition accurately, and the inaccuracies were almost completely between healthy chest scans and scans of patients with Pneumonia, which also happens to be medically justified since there is very subtle differences between the lungs of a healthy patient and a Pneumonia patient. The model does an almost perfect job at identifying patients suffering from Covid19 and Tuberculosis from images of their Chest X-Ray scans.

---

### Dataset Source:
This dataset was downloaded in raw form from the Kaggle page, [Chest X-ray Dataset - 4 Categories](https://www.kaggle.com/datasets/pritpal2873/chest-x-ray-dataset-4-categories). The image files have been left untampered with, but the folder names and filenames were not consistent in the raw folder. Therefore, I have done my own preliminary processing and renamed the folders with the *4* correct class names, namely Normal, Pneumonia, Tuberculosis, or Covid, and also sequentially organised the images within each class and renamed according to the scheme `<class name>_<image ID in 4 digits>.<file extension>` for each image, which may either be of a .jpg, .jpeg or .png format. Following this, I have split the four classes of images into training, validation and testing subsets, with an 80-10-10 percent split ratio respectively for each category. This is reflected in the repository uploaded to GitHub. The pipeline works and is on trained on the data contained within the folder 'split'.

---

# Installation instructions:
The necessary libraries have been included in the file `requirements.txt`, and can be installed in the working directory/environment by simply running the command `pip install -r requirements.txt`. The train_model() script runs on the 'split' folder, while the predict() function included in the script predict.py will run on the images found in 'data', and save the true labels versus the predictions as a CSV file in the 'logs' folder. 

---

# Other details:
All hyperprameters used in the pipeline are included in config.py, and details about the training run may be found in the 'logs' folder, including a plot of the accuracy and loss curves for both training and validation. The final performance metrics have also been saved in a csv file in the same folder, same for the details about the training run and the metrics for each epoch. Finally, the predictions obtained from running predict.py on the 'data' folder have also been saved as a CSV file in the 'logs' folder in addition to being returned as a list.

---

*Note: Any queries about the pipeline may be sent to [lokesh.venkatesh@students.iiserpune.ac.in](mailto:lokesh.venkatesh@students.iiserpune.ac.in).*