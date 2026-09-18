# Brain MRI Tumor Classification Using MobileNetV2

A deep learning project for classifying brain MRI images into three tumor categories using Transfer Learning and MobileNetV2.

## Overview

This project applies a convolutional neural network based on MobileNetV2 to classify brain MRI images into three classes:

- Glioma
- Meningioma
- Other Tumor

The goal of this project is to demonstrate the use of Transfer Learning for medical image classification in an educational and research context.

## Dataset

The dataset contains brain MRI images organized into three categories:

1. Glioma
2. Meningioma
3. Other Tumor

Images were resized to 224 × 224 pixels before being processed by the model.

The dataset was divided into training and testing subsets using an 80/20 split. During training, 10% of the training data was used for validation.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

## Model Architecture

This project uses MobileNetV2 as a pretrained feature extractor.

The original ImageNet classification head was removed using:

```python
include_top=False
