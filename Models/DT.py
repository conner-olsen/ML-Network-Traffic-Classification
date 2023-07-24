from Models.Base import AbstractModel
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.metrics import accuracy_score, \
                            confusion_matrix, \
                            precision_score, \
                            recall_score, \
                            f1_score
from Util.Util import get_results_location

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

            #**********************************************************************
    def evaluate(self, x, y, prediction):
    #
    #   Test the given model on the given data, and write the
    #       results to a file.
    #
    #   Params:
    #       x_test (DataFrame): the test data.
    #       y_test (Series): the test labels.
    #       model (DecisionTreeClassifier): the decision tree model.
    #       model_name (str): the name of the model.
    #       predictions: returned from model.predict()
    #
    #   Calls:
    #       get_result_location(models_folder, model_name)
    #
    #   Returns:
    #       Nothing, stores files
    #************************************************************
        # Compute evaluation metrics
        metrics = {
            'Accuracy': accuracy_score(y, prediction),
            'Precision': precision_score(y, prediction, average='micro'),
            'Recall': recall_score(y, prediction, average='micro'),
            'F1 Score': f1_score(y, prediction, average='micro'),
            'Confusion Matrix': confusion_matrix(y, prediction)
        }
        # Write the evaluation metrics to a file
        with open(get_results_location("DT", self.model_name), 'a') as f:
            for metric, value in metrics.items():
                f.write(f'{metric}: {value}\n')
        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f'{metric}: {value}')
