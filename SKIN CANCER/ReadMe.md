# Skin Cancer Classification Project

This project is focused on developing an AI model that classifies skin images as either **Cancer** or **Non-Cancer**. The model takes skin images as input and predicts if the skin is cancerous or not.

Check out the [**Streamlit Demo**](https://huggingface.co/spaces/metehanayhan/Skin-Cancer-Model) hosted on Hugging Face Spaces to interact with the model live.

## Overview

Skin cancer is a serious condition, and early detection is key to improving patient outcomes. This project aims to assist in this task by building a Convolutional Neural Network (CNN) model that classifies skin images into two categories:

- **Cancer**
- **Non-Cancer**

Using images of skin lesions, we train the model to recognize patterns and predict whether a given image is cancerous. The final model is deployed via Streamlit and can be accessed through Hugging Face Spaces.

## Dataset

The dataset used for this project contains two main folders:

1. **Cancer:** Images of skin lesions identified as cancerous.
2. **Non-Cancer:** Images of non-cancerous skin lesions.

The data is organized in the following format:

- `Skin_Data/Cancer/`
- `Skin_Data/Non_Cancer/`

We also encode the labels into a numerical format for model training.

## Model Architecture

We employ a **Convolutional Neural Network (CNN)** built using Keras. The architecture consists of:

- **Input layer:** Image size (170x170x3)
- **Convolutional Layers:** Feature extraction with `Conv2D` and `MaxPooling2D`
- **Dense Layers:** For classification
- **Activation Function:** `Softmax` in the final layer for binary classification.

## Results

The model was trained for **15 epochs**, achieving a validation accuracy of **91.38%**. Here are the results from the last few epochs:

| Epoch | Accuracy | Val Accuracy | Loss | Val Loss |
| --- | --- | --- | --- | --- |
| 13 | 95.10% | 91.38% | 0.1981 | 0.3473 |
| 14 | 94.83% | 86.21% | 0.1517 | 0.2525 |

## Acknowledgments

- This project uses **Hugging Face Spaces** for the Streamlit demo.
- Special thanks to the open-source community for the libraries and tools that made this project possible.

Feel free to explore the code and make contributions!