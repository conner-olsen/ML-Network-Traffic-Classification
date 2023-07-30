"""
    author: co
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 25, 2023

    This file contains utility functions for the project.
"""
import os
import pandas as pd
import time


def get_model_folder(models_folder, model_name):
    """
    Get the location of the model folder.
    If the folder does not exist, it is created.

    :param models_folder: The type of model we are training - this decides where the trained model is saved.
    :type models_folder: str
    :param model_name: The name of the file to save the model to.
    :type model_name: str
    :return: The location of the model's folder.
    :rtype: str
    """
    location = os.path.join("saved_models", models_folder, model_name)
    os.makedirs(location, exist_ok=True)
    return location


def get_results_location(models_folder, model_name):
    """
    Get the location of the results file for saving test results.
    If folder and file do not exist, they are created.

    :param models_folder: The type of model we are training - this decides where the trained model is saved.
    :type models_folder: str
    :param model_name: The name of the file to save the model to.
    :type model_name: str
    :return: The location of the result's file.
    :rtype: str
    """
    return os.path.join(get_model_folder(models_folder, model_name), "results.txt")


def get_model_location(models_folder, model_name):
    """
    Retrieves a path to a saved model from the current directory.
    Returns None if the model does not exist.

    :param models_folder: The folder containing the model.
    :type models_folder: str
    :param model_name: A sub-folder under models is named for the current tested model.
    :type model_name: str
    :return: The relative path to the model from current dir, None if model does not exist.
    :rtype: str
    """
    model_path = os.path.join(get_model_folder(models_folder, model_name), "model.pkl")
    return model_path


def load_dataset(filename):
    """
    Read dataset into DataFrame structure.

    :param filename: Name of the file to load.
    :type filename: str
    :return: DataFrame structure (pandas) of dataset.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(filename, low_memory=False)
    df.info(verbose=True)  # Print details about the dataset.
    return df


def remove_extra_columns(filename, drop_cols):
    """
    Remove a specified array of columns from a csv file.

    :param filename: CSV file to open.
    :type filename: str
    :param drop_cols: Name of column(s) to drop from csv.
    :type drop_cols: list[str]
    """
    df = pd.read_csv(filename, low_memory=False)

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv(filename, index=False)
    print("Done")
    time.sleep(5)


def load_and_prepare_data(filename, attributes):
    """
    Load the data from the given file and prepare it for supervised training or testing.

    :param filename: The name of the file to load (the dataset).
    :type filename: str
    :param attributes: The attribute names (column headers).
    :type attributes: list[str]
    :return: Preprocessed feature matrix and label vector.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    df = pd.read_csv(filename, low_memory=False)

    df["Bytes"] = pd.to_numeric(df["Bytes"], errors="coerce")
    df["Bytes"].fillna(0, inplace=True)  # Fill NaN values with 0.

    for col in attributes:
        df[col], _ = pd.factorize(df[col])

    x = df.drop(columns=["Label"])
    y = df["Label"]

    print(f"{len(df)} examples in dataset")

    return x, y
