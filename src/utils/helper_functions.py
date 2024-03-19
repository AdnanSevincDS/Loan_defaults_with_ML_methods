from typing import List, Tuple

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycorrcat.pycorrcat import corr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler


def numeric_data_summary_stats(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """
    Function that calculates a summary of statistics for numeric data 
    including count, min, max, mean, median, 25%-quant, 75%-quant, standard deviation and number of missing values.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame containing the data.
    columns (list): The list of columns in the DataFrame to be analyzed.

    Returns:
    pd.DataFrame: A DataFrame containing the statistics for the columns specified.
    """
    summary_stat_temp = df[columns].describe().T.reset_index()
    missing_dict = {}
    
    for column in columns:
        missing_dict.update({column: len(df[df[column].isnull()])})
    
    temp_missing = pd.DataFrame([missing_dict]).T.reset_index()
    temp_missing.columns = ["", "Missing"]

    summary_stat_table = pd.concat(
        [summary_stat_temp, temp_missing["Missing"]],
        axis=1).round(2)
    
    summary_stat_table = summary_stat_table[
        [
            "index",
            "count",
            "min",
            "mean",
            "50%",
            "max",
            "25%",
            "75%",
            "std",
            "Missing"
        ]]
    summary_stat_table = summary_stat_table.rename(
        columns={
            "index": " ",
            "count": "Count",
            "min": "Min",
            "mean": "Mean",
            "50%": "Median",
            "max": "Max",
            "25%": "25%-quant",
            "75%": "75%-quant",
            "std": "Standard deviation",
        })
    return summary_stat_table

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Function to plot a correlation heatmap of a DataFrame  using Kendall's method. The absolute values of the correlations are visualized,
    and the values in the upper triangle of the correlation matrix are masked.

    Parameters:
    df (pd.DataFrame): Input DataFrame on which the correlation heatmap is to be plotted.

    Returns:
    None: This function does not return any value; it only creates a plot.
    """
    corr_abs = df.corr('kendall').abs()
    mask_upper = np.triu(corr_abs)
    
    plt.figure(figsize=(16, 9))
    
    sns.heatmap(corr_abs, annot=True, fmt=".2f", cmap="coolwarm", mask=mask_upper)

    plt.show()

def numeric_multicollinearity(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Function that determines pairs of columns in the DataFrame that have a correlation exceeding the given threshold.

    Parameters:
    df (pd.DataFrame): Input DataFrame for which multicollinear pairs are to be found.
    threshold (float): The correlation coefficient threshold.

    Returns:
    pd.DataFrame: A DataFrame containing pairs of columns that exceed the correlation threshold.
    """
    corr_abs = df.corr('kendall').abs()
    corr_num_df = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool)) # Upper Mask
    corr_num_df = corr_num_df.unstack().reset_index()
    corr_num_df.columns = ['col_pair_1', 'col_pair_2', 'corr']
    corr_num_df['corr'] = corr_num_df['corr'].apply(lambda x: round(x * 100, 4))
    corr_num_df = corr_num_df[
        (corr_num_df['corr'] > threshold) &
        (corr_num_df['col_pair_1'] != corr_num_df['col_pair_2'])
    ]
    return corr_num_df.reset_index(drop=True)

def cat_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that returns a DataFrame containing each column name and its unique values and the number of unique values.

    Parameters:
    df (pd.DataFrame): Input DataFrame for which unique values per column are to be found.

    Returns:
    pd.DataFrame: A DataFrame containing the column names, their unique values and the count of these unique values.
    """
    cat_unique_df = df.apply(lambda col: col.unique()).reset_index()
    cat_unique_df.columns = ["variables", "unique_values"]
    cat_unique_df['no_unique_values'] = cat_unique_df["unique_values"].apply(lambda x: len(x))
    
    return cat_unique_df

def cat_encoding(df: pd.DataFrame) -> Tuple[list, List]:
    """
    Function that separates column names into two lists based on the number of unique values they have. 
    Columns with more than two unique values are determined for dummy encoding and those with two or fewer are for label encoding.

    Parameters:
    df (pd.DataFrame): Input DataFrame on which the encoding method needs to be decided.

    Returns:
    Tuple[list, List]:  Two lists.
    The first list contains column names for columns to be dummy encoded.
    The second list contains column names for columns to be label encoded.
    """
    cat_unique_df = cat_unique_values(df)
    get_dum = cat_unique_df[cat_unique_df["no_unique_values"] > 2]["variables"].to_list()
    label_encode = cat_unique_df[cat_unique_df["no_unique_values"] <= 2]["variables"].to_list()
    
    return get_dum, label_encode
    
def categorical_corr(df: pd.DataFrame, cat_cols: List) -> pd.DataFrame:
    """
    Function that calculates pairwise correlation coefficients between all combinations of the categorical variables using Cramer's V method.

    Parameters:
    df (pd.DataFrame): Input DataFrame on which the pairwise correlation coefficients between categorical variables need to be calculated.
    cat_cols (List): List of column names of the categorical variables.

    Returns:
    pd.DataFrame: A DataFrame containing pairs of categorical variables and their correlation coefficient.
    """
    cat_pairs = list(itertools.combinations(cat_cols, 2))
    cat_corr_values = []

    for pair1, pair2 in cat_pairs:
        cat_corr_values.append(round(
            corr_matrix(df, [pair1, pair2]).iloc[0, 1], 4)*100
        )
    
    corr_num_df = pd.DataFrame({"pairs": cat_pairs, "corr": cat_corr_values})
    
    return corr_num_df
    
def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a table that shows the number and percentage 
    of missing values for each columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame for which missing values need to be calculated.

    Returns:
    pd.DataFrame: A DataFrame containing column names, rows count for each column, 
    count and percentage of missing values per column.
    """
    missing_values = [i for i in df.isnull().sum()]
    columns = [i for i in df.columns]
    counts = [i for i in df.count()]
    
    table = pd.DataFrame()
    table["Column"] = columns
    table["DatasetRow"] = df.shape[0]
    table["ColumnRow"] = counts
    table["Missing"] = missing_values
    table["Percent"] = [round(df[i].isnull().mean() * 100, 3) for i in table["Column"]]
    
    return table.sort_values(by=['Missing'], ascending=False).reset_index(drop=True)

def fill_missing_values(
    df: pd.DataFrame,
    columns: List,
    imputation_strategy: str,
    rounding=False
) -> None:
    """
    Function that fills missing values in specified columns using a specified strategy. 
    The function supports imputation using the mean, median, mode of the column, or filling with zero. 
    It can optionally round the filling value.

    Parameters:
    df (pd.DataFrame): Input DataFrame in which missing values will be filled.
    columns (List): A list of columns in which missing values will be filled.
    imputation_strategy (str): The method to be used for imputation. 
                               Options are 'mean', 'median', 'mode' or 'zero'.
    rounding (bool): An optional argument that defaults to False. If True, the filling value will be rounded.

    Returns:
    None: This function performs the operation in-place and doesn't return anything.
    """
    for col in columns:
        if imputation_strategy == 'mean':
            filler = df[col].mean()
        elif imputation_strategy == 'median':
            filler = df[col].median()
        elif imputation_strategy == 'mode':
            filler = df[col].mode()[0]
        elif imputation_strategy == 'zero':
            filler = 0
            
        if rounding:
            filler = round(filler)
            
        df[col].fillna(filler, inplace=True)

def split_variables(df: pd.DataFrame, target: str) -> Tuple[List, List, str]:
    """
    Function that separates column names into three categories based on their types.
    The categories are categorical columns, numerical columns and the target column.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be divided into different categories.
    target (str): The target column name.

    Returns:
    List List, str: A tuple containing three elements.
    The first element is a list of names of categorical columns.
    The second element is a list of names of numerical columns.
    The third element is the name of the target column.
    """
    cat_cols = df.dtypes[df.dtypes == "object"].index.to_list()
    num_cols = df.dtypes[df.dtypes != "object"].index.to_list()
    target_cols = target
    num_cols.remove(target_cols)

    return cat_cols, num_cols, target_cols

def auc_gini_score(model, y_data, x_data) -> Tuple[int, int]:
    """
    Function that calculates the Area Under the Receiver Operating Characteristics Curve (AUC-ROC) 
    and the Gini coefficient for the predictions of a model.

    Parameters:
    model:  The trained model for which AUC-ROC and Gini scores are to be calculated.
    y_data : The true labels.
    x_data : The input data to be predicted by the model.

    Returns:
    float: AUC-ROC and Gini scores.
    """
    auc = roc_auc_score(y_data, model.predict(x_data))
    gini = 2 * auc - 1

    return auc, gini

def scale_and_prepare_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function that scales feature data using a StandardScaler, making the mean of each feature equal to 0 and 
    the variance equal to 1. It also reshapes the target data to be one-dimensional.

    Parameters:
    X_train (pd.DataFrame): The training feature data.
    X_test (pd.DataFrame): The test feature data.
    y_train (pd.DataFrame): The training target data.
    y_test (pd.DataFrame): The test target data.

    Returns:
    pd.DataFrame : The scaled training
    pd.DataFrame: The scaled test
    pd.DataFrame: The reshaped training target data
    pd.DataFrame: The reshaped target data.
    """
    scaler = StandardScaler()

    # Reshape y_train and y_test to be one-dimensional
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Fit the scaler to the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Assign scaled to X_train and X_test
    X_train = X_train_scaled.copy()
    X_test = X_test_scaled.copy()

    return X_train, X_test, y_train, y_test

def random_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    random_state=42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that applies Random Under Sampling (RUS) to balance the classes in the training data.

    Parameters:
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.DataFrame): The training target data.
    random_state (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The resampled feature and target data.
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)  

    # Assign resamples to trains dataset
    X_train = X_resampled.copy()
    y_train = y_resampled.copy()

    return X_train, y_train

def run_grid_search(model, param_grid, cv=3, verbose=1, scoring='roc_auc', n_jobs=-1):
    """
    Function that runs grid search cross-validation for hyperparameter tuning on a model.

    Parameters:
    model : The model for which hyperparameters are to be tuned.
    param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    cv (int): Determines the cross-validation splitting strategy. Defaults to 3.
    verbose (int): Controls the verbosity: the higher, the more messages. Default is 1.
    scoring (str): A single string to evaluate the predictions on the test set.. Defaults to 'roc_auc'.
    n_jobs (int): The number of jobs to run in parallel. '-1' means using all processors. Default is -1.

    Returns:
     The grid search object after it has been fit.
    """
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=cv, 
                        verbose=verbose,
                        scoring=scoring,
                        n_jobs=n_jobs)
    return grid

def get_output(model, y_train: np.ndarray, X_train: np.ndarray, y_test: np.ndarray, X_test: np.ndarray) -> pd.DataFrame:
    """
    Function that calculates the AUC-ROC and Gini scores on both the training and test datasets for a trained model.

    Parameters:
    model : The trained model for which the scores are to be calculated.
    y_train : The true labels for the training data.
    X_train : The training data.
    y_test :  The true labels for the test data.
    X_test : The test data.

    Returns:
    pd.DataFrame: A DataFrame containing the training and test AUC-ROC and Gini scores.
    """
    result_dict = {
        'Train_AUCROC': [auc_gini_score(model, y_train, X_train)[0]],
        'Test_AUCROC': [auc_gini_score(model, y_test, X_test)[0]],
        'Train_Gini': [auc_gini_score(model, y_train, X_train)[1]],
        'Test_Gini': [auc_gini_score(model, y_test, X_test)[1]],
    }

    return pd.DataFrame(result_dict)


def get_hyper_output(model, y_train: np.ndarray, X_train: np.ndarray, y_test: np.ndarray, X_test: np.ndarray) -> pd.DataFrame:
    """
    Function that calculates the AUC-ROC and Gini scores on both the training and test datasets for a trained model.

    Parameters:
    model : The trained model for which the scores are to be calculated.
    y_train : The true labels for the training data.
    X_train : The training data.
    y_test : The true labels for the test data.
    X_test : The test data.

    Returns:
    pd.DataFrame: A DataFrame containing the training and test AUC-ROC and Gini scores.
    """
    result_dict = {
        'Best Parameters': [str(model.best_params_)],
        'Train_AUCROC': [auc_gini_score(model, y_train, X_train)[0]],
        'Test_AUCROC': [auc_gini_score(model, y_test, X_test)[0]],
        'Train_Gini': [auc_gini_score(model, y_train, X_train)[1]],
        'Test_Gini': [auc_gini_score(model, y_test, X_test)[1]],
    }

    return pd.DataFrame(result_dict)

def class_report_and_cm(y_test: np.ndarray, pred: np.ndarray, model):
    """
    Function that prints the classification report and confusion matrix, 
    and plots the confusion matrix for the predictions of a model.

    Parameters:
    y_test : The true labels for the test data.
    pred : The predicted labels for the test data.
    model: The model that made the predictions.

    Returns:
    None: This function does not return anything; it prints and plots the results.
    """
    print("Classification report:\n")
    print(classification_report(y_test, pred))
    
    print("Display confusion matrix:\n")
    cm = confusion_matrix(y_test, pred)
    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()