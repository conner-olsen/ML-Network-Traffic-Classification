"""
    author: co
    project: group 7 term project
    class: CS-534 Artificial Intelligence WPI
    date: July 20, 2023,
    last update: July 25, 2023

    class to act as a base for all models
"""
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from Util.Util import get_results_location


class AbstractModel(ABC):
    """
    Abstract class for models.
    """

    @abstractmethod
    def __init__(self, model_name: str = "default"):
        """
        Constructor.
        """
        self.model_name = model_name
        self.model = None
        self.model_type = None
        self.attributes = [
            "Duration",
            "Src_IP",
            "Src_Pt",
            "Dst_Pt",
            "Dst_IP",
            "Packets",
            "Bytes",
            "Flags",
            "Label",
        ]

    def prepare_data(self, df, num_records=None):
        """
        Prepare data for training.
        This includes:
            - Selecting only the attributes we want to use.
            - Converting the Flags column to numeric.
            - Selecting only the first num_records records.

        :param num_records: Number of records to use from the dataset.
        :param df: "Data frame" Data loaded from csv.
        :type df: pandas.DataFrame
        :return: tuple of feature matrix and label vector.
        :rtype: tuple[pandas.DataFrame, pandas.Series]
        """
        if self.attributes is not None:
            df = df[self.attributes]

        if num_records is not None:
            df = df.head(num_records)

        # Convert Flags column to numeric via factorize.
        df["Flags"] = pd.factorize(df["Flags"])[0]

        x = df.drop(columns=["Label"])
        y = df["Label"]

        print(f"{len(df)} examples in dataset")

        return x, y

    def set_model(self, trained_model):
        """
        After training the model, update the object to the trained data

        :param trained_model: Trained model
        :type trained_model: sklearn.BaseEstimator
        """
        self.model = trained_model

    def get_model_name(self):
        """
        Get model name

        :return: Model name
        :rtype: str
        """
        return self.model_name

    def train_model(self, x, y):
        """
        Fit wrapper function

        :param x: Data
        :type x: pandas.DataFrame
        :param y: Labels
        :type y: pandas.Series
        """
        self.model.fit(x, y)

    def test_model(self, x):
        """
        Test model

        :param x: Test data
        :type x: pandas.DataFrame
        :return: Predictions
        :rtype: numpy.array
        """
        return self.model.predict(x)

    def evaluate(self, y_test, predictions):
        """
        Test the given model on the given data, and write the
        results to a file.

        :param y_test: Labels
        :param predictions: returned from model.predict()
        :return: Nothing
        """

        # Compute evaluation metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="micro"),
            "Recall": recall_score(y_test, predictions, average="micro"),
            "F1 Score": f1_score(y_test, predictions, average="micro"),
            "Confusion Matrix": confusion_matrix(y_test, predictions),
        }

        # Write the evaluation metrics to a file
        with open(get_results_location(self.model_type, self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    @abstractmethod
    def render_model(self, x, y):
        """
        Render the model
        :param x: the input parameters
        :param y: the calculated output
        :return: None
        """
        pass
