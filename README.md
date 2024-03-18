#### This material may not be copied or published elsewhere (including Facebook and other social media) without the permission of the author!

# A Short Summary

This study covers the use of machine learning methods in predicting default in mortgage loans.
Lending institutions can leverage machine learning methods to handle loan applications and
prevent losses from defaulted loans. This study assesses the predictive performance of various
machine learning methods using the Freddie Mac Single-Family Loan-Level Dataset, focusing
on loans that originated in 2007. 

We employed widely used machine learning methods, including logistic regression, decision tree, random forest and XGBoost to predict defaults in mortgage loans together with under-sampling technique implemented to alleviate the impact of the imbalanced dataset
problem. 

We found that the XGBoost method combined with the under-sampling technique yielded the highest AUC (Area under the ROC Curve) score. 

| Model               | Without hyperparameter | with hyperparameter | Search for Hyperparameter                                                                             |
|---------------------|------------------------|---------------------|-------------------------------------------------------------------------------------------------------|
| Logistic Regression | 75.9007%               | 75.9007%            | c>[0.01, 0.1, 1, 10, 50] penalty>['none','l1','l2'] solver>['lbfgs', 'sag','saga','newton-cg']        |
| Decision Tree       | 67.015%                | 73.551%             | criterion >['gini','entropy'] min_samples_split > [2,3,5,10] max_depth >[None,4, 5, 6, 8]             |
| Random Forest       | 75.872%                | 75.958%             | n_estimators >[50,100, 200] max_depth >[None,4, 6, 8]                                                 |
| XGBoost             | 76.040                 | 76.339%             | n_estimators>[50,100, 200] max_depth> [3, 6,10] learning_rate> [0.05,0.1,0.5] subsample > [0.5,0.7,1] |


## Notebooks Directory
This section includes three distinct Jupyter notebooks that perform different aspects of data analysis:

```
├── data_modeling.ipynb: This notebook is utilized for exploratory data analysis (EDA) and initial modeling of the data.

├── algorithms.ipynb: This notebook implements machine learning models such as Linear Regression, Decision Tree, Random Forest and XGBoost.

├── algorithms_tuning.ipynb: This notebook is focused on fine-tuning the hyperparameters of the models for optimal performance.

├── src_demo.ipynb: This notebook checks whether setup.py has been correctly set up.
│   ├── from src.models import train_model has been tested.
│   ├── from src.data import make_dataset  has been tested.
│   ├── from src.tests import test_make_dataset has been tested.
│   ├── from src.models import predict_model  has been tested.
│   

```

Please navigate to each directory for a more detailed observation and understanding of how the project flows and functions.



Mortgage Loan Defaults with Machine Learning Methods
==============================

Machine learning methods in predicting default in mortgage loans

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── entities       <- Scripts for creating dataclasses 
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


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

**Version**: 1.0