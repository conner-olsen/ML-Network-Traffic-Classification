from Models.Base import AbstractModel
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from Util.Util import get_results_location


class DT(AbstractModel):
    def __init__(self, model_name: str = "default"):
        """
        Decision tree constructor
        Note: More additional parameters should be used for the classifier

        :param model_name: Target SVM
        :type model_name: str
        """
        self.model_name = model_name
        self.model = DecisionTreeClassifier()

    def render_model(self, model, x_train, y_train):
        """
        Visual representation of the model, save as .png

        :param model: the decision tree model
        :type model: DecisionTreeClassifier
        :param x_train: the training data
        :type x_train: DataFrame
        :param y_train: the training labels
        :type y_train: Series
        :returns: nothing, displays tree and saves to model type folder
        """
        # Load and preprocess the data
        feature_names = x_train.columns.tolist()
        class_names = sorted(y_train.unique().astype(str))

        # Create and render the decision tree
        dot_data = tree.export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
        )
        graph = graphviz.Source(dot_data, format="png")
        graph.render("DT/" + self.model_name + "/_dt", view=True)

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
        with open(get_results_location("DT", self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
