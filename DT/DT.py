import graphviz
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


class DT:
    def __init__(self, model_name: str = 'default'):
        """
        Decision tree constructor
        Note: More additional parameters should be used for the classifier

        :param model_name: Target SVM
        """
        self.model_name = model_name
        self.model = DecisionTreeClassifier()

    def set_model(self, trained_model):
        """
        After training the model, update the object to the trained data

        :param trained_model: Trained model
        :return: Nothing
        """
        self.model = trained_model

    def get_model_name(self):
        """
        Get model name

        :return: Model name
        """
        return self.model_name

    def train_model(self, x, y):
        """
        Fit wrapper function

        :param x: Data
        :param y: Labels
        :return: Trained model
        """
        return self.model.fit(x, y)

    def test_model(self, x):
        """
        Test DT model

        :param x: Test data
        :return: Return from SVC predict function
        """

        return self.model.predict(x)

    # *********************************************************
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


def prepare_data(df, attributes, length=-1):
    """
    Prepare data for training

    Note: May be irrelevant, @Corey adjust as needed for the changes

    :param df: "Data frame" Data loaded from csv
    :param attributes: Data headers to normalize into integer values
    :param length: Length of data to prepare
    :return:
        x, y (tuple):
            x = Samples
            Y = Labels
    """
    if length > 0:
        df = df.iloc[:length]

    # Convert 'Bytes' column to numeric, setting non-numeric values to NaN
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors="coerce")
    df['Bytes'] = df['Bytes'].fillna(0)  # Fill NaN values with 0

    # Factorize categorical features to integer labels
    for col in attributes:
        df[col], _ = pd.factorize(df[col])

    x = df.drop(columns=['Label'])
    y = df['Label']

    # Debug print
    print(str(len(df)) + " examples in dataset")

    return x, y
