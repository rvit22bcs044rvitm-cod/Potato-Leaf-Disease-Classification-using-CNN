##Project Title
Potato Leaf Disease Detection using Image Classification

##Introduction
This project focuses on building an image classification model to detect and classify potato plant diseases. The dataset consists of high resolution images of potato leaves categorized into three classes: Early Blight, Late Blight, and Healthy leaves. The goal is to develop a robust deep learning model that can accurately identify plant diseases and support agricultural diagnostics.

##Dataset Details
The dataset contains three primary classes of potato leaves:

Early Blight

Late Blight

Healthy

The images are high resolution and were processed in a Google Colab environment using a zipped dataset structure for efficient loading.

##Objectives

Understand image classification using deep learning.

Perform data preprocessing including resizing and normalization.

Apply data augmentation techniques like flipping and rotation.

Train and evaluate a Convolutional Neural Network (CNN).

###Technical Workflow
The project follows a standard machine learning pipeline:

Data Extraction: Unpacking the zipped dataset in the Colab environment.

Data Loading: Categorizing images from directories into training, validation, and testing sets.

Preprocessing: Resizing images to 256 by 256 pixels and scaling pixel values between 0 and 1.

Model Building: Constructing a CNN with convolutional layers for feature extraction and pooling layers for dimension reduction.

Training: Running the model for 20 epochs using the Adam optimizer.

Evaluation: Measuring performance using accuracy, confusion matrices, and classification reports.

###Results
The model achieved an overall test accuracy of 95 percent.

Early Blight detection was highly successful with nearly perfect precision.

Late Blight detection showed strong results but revealed some confusion with healthy leaves.

The Healthy class showed lower precision due to a smaller number of samples in the original dataset compared to the disease classes.

###Conclusion
The CNN model successfully identifies potato leaf diseases with high reliability. To improve the model further, additional healthy leaf images should be added to balance the dataset and reduce false negative errors in the Late Blight category.

####How to Use

Upload the potato leaf dataset as a zip file to the Google Colab environment.

Run the provided notebook cell to extract data and start the training process.

Observe the accuracy and loss plots to verify model convergence.

Review the confusion matrix for a detailed breakdown of classification performance.
