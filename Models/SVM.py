from abc import ABC

from Models.Base import AbstractModel
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from Util.Util import get_results_location


class SVM(AbstractModel, ABC):
    def __init__(
        self,
        attributes,
        k_type,
        d=3,
        c=0.0,
        v=True,
        cache=8000,
        model_name: str = "default",
    ):
        """
        Constructor to initialize the SVM class

        :param attributes: list of column headers to use for training and testing (may not handle full column dimension)
        :type attributes: list of str
        :param k_type: kernel type - poly, rbf, linear, etc.
        :type k_type: str
        :param d: dimension (for poly kernel)
        :type d: int
        :param c: regularization parameter
        :type c: float
        :param v: verbose (good for debugging)
        :type v: bool
        :param cache: cache size (bytes)
        :type cache: int
        :param model_name: name of the SVM being trained
        :type model_name: str
        """
        self.model_name = model_name
        self.attributes = attributes
        self.model = SVC(
            kernel=k_type, degree=d, C=c, verbose=v, cache_size=cache
        )  # additional params to init SVC

    def evaluate(self, x, y, prediction):
        """
        Test the given model on the given data, and write the
        results to a file.

        :param x: the test data
        :type x: DataFrame
        :param y: the test labels
        :type y: Series
        :param prediction: returned from model.predict()
        :type prediction: array-like
        :returns: Nothing, stores files
        """
        # Compute evaluation metrics
        metrics = {
            "Accuracy": accuracy_score(y, prediction),
            "Precision": precision_score(y, prediction, average="micro"),
            "Recall": recall_score(y, prediction, average="micro"),
            "F1 Score": f1_score(y, prediction, average="micro"),
            "Confusion Matrix": confusion_matrix(y, prediction),
        }
        # Write the evaluation metrics to a file
        with open(get_results_location("SVM", self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
