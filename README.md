# NBA MVP Predictor
NBA MVP Predictor
This Python script predicts the Most Valuable Player (MVP) for the NBA (National Basketball Association) based on player statistics. It utilizes machine learning models, specifically XGBoost and Random Forest, to make predictions for MVP candidates.

Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [2023-2024 Predictions](#2023-2024-predictions)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
  
Introduction
The goal of this project is to predict the NBA MVP for a given season based on player performance statistics. The script uses machine learning techniques, specifically XGBoost and Random Forest, to make predictions. The models are trained on historical data, and their accuracy is evaluated against actual MVP winners.

## Data
The dataset used for training and prediction is sourced from Kaggle: NBA Player Season Statistics with MVP Win Share. The dataset contains various player statistics for NBA seasons, including information about MVP winners.

## Data Preprocessing
The script performs data preprocessing tasks, including handling null values, removing redundant columns, and filtering players based on specific criteria. The dataset is split into players who received MVP votes and those who did not, allowing for a more focused analysis.

## Feature Selection
To identify the most important features for MVP prediction, the script employs a Random Forest classifier. The top features include metrics such as Win Shares per 48, Win Shares, Box Plus Minus, Offensive Win Shares, and Margin of Victory (adjusted).

## Model Training
Two machine learning models, XGBoost and Random Forest, are trained on the resampled dataset. The resampling is done using Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.

## Model Evaluation
The models are evaluated using Mean Absolute Error (MAE) on test data for each NBA season. The script provides insights into the accuracy of the models and their predictions for MVP winners.

## 2023-2024 Predictions
The final section of the script focuses on making predictions for the 2023-2024 NBA season. It imports recent player and team statistics, preprocesses the data, and utilizes the trained XGBoost model to predict the MVP winner and top candidates.

## Dependencies
The script requires the following Python libraries:

pandas
numpy <br>
matplotlib <br>
seaborn <br>
scikit-learn <br>
imbalanced-learn <br>
xgboost <br>
requests <br>
BeautifulSoup <br>
Install these dependencies using pip install <library> <br>

## How to Use
Ensure all dependencies are installed.
Run the script in a Python environment.
The script will perform data preprocessing, model training, and make predictions for the specified NBA season.
Feel free to modify the script for different datasets or customize the prediction logic.
