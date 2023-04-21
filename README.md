# Classification-with-a-Tabular-Vector-Borne-Disease-Dataset
# Vector Borne Disease Prediction

This repository contains the code and data for predicting vector-borne diseases using machine learning and deep learning models in R.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Models](#models)
  - [Machine Learning Model](#machine-learning-model)
  - [Deep Learning Model](#deep-learning-model)
- [Usage](#usage)
- [Results](#results)

## Getting Started

To get started, clone this repository and install the required R packages:

```R
install.packages("randomForest")
install.packages("keras")
install.packages("tidyverse")
```

# Dataset
The dataset used in this project consists of a train set and a test set with features related to vector-borne diseases. Download the dataset from the following links:

Train dataset: train.csv
Test dataset: test.csv

# Models
Machine Learning Model
We have implemented a Random Forest model using the randomForest package in R. The model is trained on the preprocessed dataset and is used to make predictions on the test dataset.

# Deep Learning Model
We have also implemented a deep learning model using the keras package in R. The model consists of a sequential architecture with multiple dense layers and is trained on the preprocessed dataset. This model is also used to make predictions on the test dataset.

# Usage
Follow the steps below to train and evaluate the models:

# Load and preprocess the dataset.
Train the machine learning model (Random Forest) and deep learning model (Keras).
Make predictions on the test dataset using both models.
Visualize the predictions and save them to separate submission files.

# Results
The results of the models are saved as separate submission files:
  Random Forest predictions: submission_rf.csv
  Keras predictions: submission_keras.csv
Additionally, the distribution of predictions for both models is visualized using bar plots.
