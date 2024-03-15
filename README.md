#### This repository was established as a case study for a potential Junior/Middle DS position.
#### This material may not be copied or published elsewhere (including Facebook and other social media) without the permission of the author!

# A Short Summary

This study covers the use of machine learning methods in predicting default in mortgage loans.
Lending institutions can leverage machine learning methods to handle loan applications and
prevent losses from defaulted loans. This thesis assesses the predictive performance of various
machine learning methods using the Freddie Mac Single-Family Loan-Level Dataset, focusing
on loans that originated in 2006 and 2007. 

We employed widely used machine learning methods, including logistic regression, decision tree, random forest and XGBoost to predict defaults in mortgage loans together with under-sampling technique implemented to alleviate the impact of the imbalanced dataset
problem. 

We found that the XGBoost method combined with the under-sampling technique yielded the highest AUC (Area under the ROC Curve) score. 

The important variables in the prediction of default obtained with help of XGBoost please look at the featureImportance.ipynb

# Repository File Structure and Description 

This repository contains the following key components:

```
project-root
│
├── dataset
│   ├── rawdata.csv                           # Raw Dataset
│   ├── X_test.csv                            # Split Dataset for Testing Features
│   ├── X_train.csv                           # Split Dataset for Training Features
│   ├── y_test.csv                            # Split Dataset for Testing Labels
│   └── y_train.csv                           # Split Dataset for Training Labels
│
└── notebooks
    ├── dataModeling.ipyb                     # Notebook 1: Data Modeling
    ├── algorithms.ipynb                      # Notebook 2: Algorithms
    └── algorithmsTuning.ipynb                # Notebook 3: Algorithms Tuning
    └── featureImportance.ipynb               # Notebook 4: Feature Importance

```

## Dataset Directory 
This section includes both the original raw dataset and the segmented or split data files. Here's what each file represents:

```
- rawdata.csv: This original, unprocessed dataset.
- X_test.csv
- X_train.csv
- y_test.csv
- y_train.csv

```

## Notebooks Directory
Contained in this division are three distinct Jupyter notebooks that perform different aspects of data analysis:

```
- dataModeling.ipynb: This notebook is utilized for exploratory data analysis (EDA) and initial modeling of the data.

- algorithms.ipynb: This notebook implements machine learning models such as Linear Regression, Decision Tree, Random Forest and XGBoost.

- algorithmsTuning.ipynb: This notebook is focused on fine-tuning the hyperparameters of the models for optimal performance.

```

Please navigate to each directory for a more detailed observation and understanding of how the project flows and functions.


# Data Description & Preparation


• Non-Default: The loan is deemed non-defaulted if its delinquency status is below
90 days throughout the 24-month window

• Default: The loan is deemed defaulted if its delinquency status at least once
exceeds 90 days throughout the 24-month window.

The independent variables used in this study were selected from the origination dataset.
In contrast, the dependent variable was generated from the current loan delinquency status
variable in the monthly performance dataset based on the adopted definition of default. None
of the variables in the dataset have more than 50% missing values. In order to deal with the
missing values, a common approach was used where the missing values were replaced by their
mean value during data preparation. One-hot encoding was applied to categorical variables to
convert them into numerical dummy variables, enabling them to use in the modeling process.
The dataset was divided into a training set and a test set with 70% and 30% ratios, respectively.
The models were trained only on the training set, and their performance was evaluated on the
test set, which was an unseen dataset during the training process. 

Based on the adopted definition of default, dataset consists of 37,769 defaulted loans, and suffers
an imbalance dataset problem, with 6.00% defaulted ratios. The defaulted loans represent the minority class.

Characteristics of the dataset:

| #Samples | #Variables | #Default       | #Non-Default     |
|----------|------------|----------------|------------------|
| 629,544  | 17         | 37,769 (6.00%) | 591,775 (94.00%) |

**Author**:Adnan Sevinc

**Date**: 15.03.2024

**Version**: Draft

