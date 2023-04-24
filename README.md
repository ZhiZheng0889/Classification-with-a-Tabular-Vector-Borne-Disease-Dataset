# Classification-with-a-Tabular-Vector-Borne-Disease-Dataset
# Vector Borne Disease Prediction

This repository contains the code and data for predicting vector-borne diseases using machine learning models, including Random Forest, XGBoost, and deep learning models in R.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Models](#models)
  - [Random Forest Model](#random-forest-model)
  - [XGBoost Model](#xgboost-model)
  - [Deep Learning Model](#deep-learning-model)
- [Usage](#usage)
- [Results](#results)

## Getting Started

To get started, clone this repository and install the required R packages:

```R
install.packages("randomForest")
install.packages("xgboost")
install.packages("keras")
install.packages("tidyverse")
```

## Dataset
The dataset used in this project consists of a train set and a test set with features related to vector-borne diseases. Download the dataset from the following links:
https://www.kaggle.com/datasets/richardbernat/vector-borne-disease-prediction
Train dataset: train.csv
Test dataset: test.csv

## Models
### Random Forest Model
We have implemented a Random Forest model using the randomForest package in R. The model is trained on the preprocessed dataset and is used to make predictions on the test dataset. Hyperparameter tuning is performed for improving the model's performance.

### XGBoost Model
We have implemented an XGBoost model using the xgboost package in R. The model is trained on the preprocessed dataset and is used to make predictions on the test dataset. Hyperparameter tuning is performed for improving the model's performance.

### Deep Learning Model
We have also implemented a deep learning model using the keras package in R. The model consists of a sequential architecture with multiple dense layers and is trained on the preprocessed dataset. This model is also used to make predictions on the test dataset.

## Usage
Follow the steps below to train and evaluate the models:

1. Load and preprocess the dataset.
2. Train the Random Forest model and perform hyperparameter tuning.
3. Train the XGBoost model and perform hyperparameter tuning.
4. Train the deep learning model (Keras).
5. Make predictions on the test dataset using all three models.
6. Visualize the predictions and save them to separate submission files.

## Results
The results of the models are saved as separate submission files:

Random Forest predictions: submission_rf.csv
XGBoost predictions: submission_xgb.csv
Keras predictions: submission_keras.csv
Additionally, the distribution of predictions for all models is visualized using bar plots.
