# KM.py

"""
    :author: EM
    :project: Group 7
    :class: CS534 Artificial Intelligence WPI
    :date: July 23, 2023

    An alternative class to FKM - FKM is very complicated and may
    be more of a long-term project. K-means has a sklearn library
    already, while FKM is being written from scratch. Decision to impl.
    This to fulfill project requirements, but hoping to continue work with
    FKM for future research project paper, not necessarily on the same dataset.

    API ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict
"""

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
    silhouette_score,
    adjusted_mutual_info_score,
    completeness_score,
)

from Models.Base import AbstractModel
from Util.Util import get_results_location


class KM(AbstractModel):
    VERBOSE = 0  # global verbose (passed to model so we have status msg in dev.)

    def __init__(self, k=8, model_name="default"):
        """
        K-Means class constructor - initialize model, set k

        :param k: number of clusters (default is 8)
        :type k: int
        :param model_name: name the FM models
        :type model_name: str
        """
        self.k = k
        self.model_name = model_name
        self.model = KMeans(
            n_clusters=self.k, algorithm="lloyd", random_state=0, verbose=self.VERBOSE
        )

    def prepare_data(self, df):
        """
        Additional data processing needed for a specific model, pass both
        train and test df's to split into samples and targets

        :param df: The whole, pre-processed dataframe (samples and target)
        :type df: dataframe
        :returns: X (dataframe): the samples (train or test), y (dataframe): the targets (train or test)
        """
        X = df.drop(columns=["Label"])
        y = df["Label"]

        return X, y

    def train_model(self, x_train, y_train):
        """
        Train the model - fit to X train samples

        :param x_train: the training dataset samples
        :type x_train: dataframe
        :param y_train: None, retained for API consistency
        :type y_train: dataframe
        :returns: trained model
        """
        self.model = self.model.fit(x_train)
        return self.model

    def test_model(self, x_test):
        """
        Test the model - the "predict" function on fitted model and test
        data

        :param x_test: the testing dataset samples
        :type x_test: dataframe
        :returns: predictions from "predict" function call
        """
        return self.model.predict(x_test)

    def evaluate(self, x, y, predict):
        """
        Evaluate clustering accuracy

        :param x: samples
        :type x: dataframe
        :param y: targets from a training set or testing set
        :type y: dataframe
        :param predict: predictions from model
        :type predict: array-like
        :returns: nothing, prints to result file

        API ref:
        https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        """
        # Compute evaluation metrics
        metrics = {
            "Random Index": rand_score(y, predict),
            "Adjusted Random Index": adjusted_rand_score(y, predict),
            "Silhouette": silhouette_score(x, predict),  # takes very long on lg data
            "Adjusted Mutual Info": adjusted_mutual_info_score(y, predict),
            "Completeness": completeness_score(y, predict),
        }

        # Write the evaluation metrics to a file
        with open(get_results_location("KM", self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    def render_model(self, model, x, y):
        """
        Visualize the k clusters

        :param model: Fitted model
        :type model: sklearn.cluster.KMeans
        :param x: dataset samples
        :type x: dataframe
        :param y: dataset targets (unused, kept for API consisitency)
        :type y: dataframe
        :returns: nothing, but displays and saves a .png file of cluster plot
        """

        og = x[model == 0]  # indexes

        # this does not really represent the relation of the clusters well
        plt.scatter(og.iloc[:, 0], og.iloc[:, 1], color="black")
        plt.scatter(model, y, color="red")

        plt.show()
