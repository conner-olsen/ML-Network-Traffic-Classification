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
import warnings
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
    accuracy_score,
)

from Models.Base import AbstractModel
from Util.Util import get_results_location


class KM(AbstractModel):
    VERBOSE = False  # global verbose (passed to model so we have status msg in dev.)

    def __init__(self, k=8, model_name="default"):
        """
        K-Means class constructor - initialize model, set k

        :param k: number of clusters (default is 8)
        :type k: int
        :param model_name: name the FM models
        :type model_name: str
        """
        super().__init__()
        self.k = k
        self.model_name = model_name
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.model = KMeans(
            n_clusters=self.k, algorithm="elkan", random_state=0, verbose=self.VERBOSE
        )
        self.model_type = "KM"

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
            "Homogeneity": homogeneity_score(y_test, predictions),
            "Completeness": completeness_score(y_test, predictions),
            "V_Measure": v_measure_score(y_test, predictions),
            "Accuracy": accuracy_score(y_test, predictions),
        }

        # Write the evaluation metrics to a file
        with open(get_results_location(self.model_type, self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    def render_model(self, x, y):
        """
        Visualize the k clusters

        :param x: dataset samples
        :type x: dataframe
        :param y: dataset targets (unused, kept for API consistency)
        :type y: dataframe
        :returns: nothing, but displays and saves a .png file of cluster plot
        """

        # Plot the data points
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=self.model.labels_, cmap="viridis")

        # Plot the cluster centers
        centers = self.model.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c="red", s=300, alpha=0.5)

        plt.show()

    def train_model(self, x, y):
        """
        Train the model on the given data

        :param x: dataset samples
        :type x: dataframe
        :param y: dataset targets
        :type y: dataframe
        :return: Nothing
        """
        self.model.fit(x)
