"""
    author: co
    project: group 7 term project
    class: CS-534 Artificial Intelligence WPI
    date: July 5, 2023,
    last update: July 25, 2023

    class to implement the decision tree model
"""
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from Models.Base import AbstractModel
from Util.Util import get_model_folder


class DT(AbstractModel):
    def __init__(self, model_name: str = "default"):
        """
        Decision tree constructor
        Note: More additional parameters should be used for the classifier

        :param model_name: Target SVM
        :type model_name: str
        """
        super().__init__()
        self.model_name = model_name
        self.model = DecisionTreeClassifier()
        self.model_type = "DT"

    def render_model(self, x_train, y_train):
        """
        Visual representation of the model, saved as a .png file.

        :param x_train: The training data.
        :type x_train: DataFrame
        :param y_train: The training labels.
        :type y_train: Series
        :returns: Nothing, displays a tree and saves it to the model type folder.
        """
        feature_names = x_train.columns.tolist()
        class_names = sorted(y_train.unique().astype(str))

        dot_data = tree.export_graphviz(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
        )
        graph = graphviz.Source(dot_data, format="png")
        graph.render(
            get_model_folder(self.model_type, self.model_name) + "/_dt", view=True
        )
        print("Decision tree rendered and saved to DT/" + self.model_name + "/_dt.png")
