#### This repository was established as a case study for a potential Junior/Middle DS position.
#### This material may not be copied or published elsewhere (including Facebook and other social media) without the permission of the author!

# Repository File Structure and Description 

This repository contains the following key components:

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

## Dataset Directory 
This section includes both the original raw dataset and the segmented or split data files. Here's what each file represents:

- **rawdata.csv**: This original, unprocessed dataset.
- **X_test.csv**
- **X_train.csv**
- **y_test.csv**
- **y_train.csv**

## Notebooks Directory
Contained in this division are three distinct Jupyter notebooks that perform different aspects of data analysis:

- **dataModeling.ipynb**: This notebook is utilized for exploratory data analysis (EDA) and initial modeling of the data.

- **algorithms.ipynb**: This notebook implements machine learning models such as Linear Regression (LR) and XGBoost.

- **algorithmsTuning.ipynb**: This notebook is focused on fine-tuning the hyperparameters of the models for optimal performance.

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

