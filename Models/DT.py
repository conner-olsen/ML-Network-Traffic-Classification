from Models.Base import AbstractModel
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz


class DT(AbstractModel):
    def __init__(self, model_name: str = 'default'):
        """
        Decision tree constructor
        Note: More additional parameters should be used for the classifier

        :param model_name: Target SVM
        """
        self.model_name = model_name
        self.model = DecisionTreeClassifier()

    def render_model(self, model, x_train, y_train):
        #
        #   visual representation of the model, save as .png
        #
        #   Params:
        #       model (DecisionTreeClassifier): the decision tree model.
        #       x_train (DataFrame): the training data
        #       y_train (Series): the training labels
        #
        #   Returns:
        #       nothing, displays tree and saves to model type folder
        # *********************************************************
        # Load and preprocess the data
        feature_names = x_train.columns.tolist()
        class_names = sorted(y_train.unique().astype(str))

        # Create and render the decision tree
        dot_data = tree.export_graphviz(model, out_file=None, feature_names=feature_names,
                                        class_names=class_names, filled=True)
        graph = graphviz.Source(dot_data, format="png")
        graph.render("DT/" + self.model_name + "/_dt", view=True)
