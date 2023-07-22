#SVM.py

'''
    author: co, em
    project: group 7 
    class: CS-534 WPI
    last update: July 5, 2023

    Class file for the SVM implementation

    API reference: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
'''
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DT:

    #**********************************************************
    def __init__(self, model_name: str='default'):
    #
    #   constructor
    #   
    #   params:
    #       model_name (str): name of the SVM being trained
    #   
    #   TODO:
    #       any initial values for DT Classifier?
    #**********************************************************
        self.model_name = model_name
        self.model = DecisionTreeClassifier()   #additional params to init tbd

    #**********************************************************
    def set_model(self, trained_model):
    #
    #    After training the model, update the model object to 
    #        the trained model
    #
    #    parameters:
    #        trained_model: model after training
    #    returns:
    #        nothing, set function
    #***********************************************************
        self.model = trained_model

    
    #**********************************************************
    def get_model_name(self):
    #
    #   function to return the model name
    #
    #   returns:
    #       model_name (str): name of model in training
    #**********************************************************
        return self.model_name

    
    #************************************************************
    def prepare_data(self, df, attributes, length=-1):
    #
    #   function to prepare data for training Decision Tree
    #
    #   params:
    #       df (DataFrame): data as loaded from .csv file
    #       attributes (str[]): catagorical data column headers that 
    #           should be normalized (turned into integer values)
    #       length: send a length to prepare just a subset of the dataset
    #   returns:
    #       x, y (tuples): 
    #           x = samples (data with no labels column)
    #           y = labels column only  
    #
    #   TODO: whatever changes were made to data already may 
    #       make much of this irrelevent - see conner's comment on
    #       discord - also @Corey pls adjust this as needed with 
    #       your dataset changes. 
    #*************************************************************
        if (length > 0):
            df = df.iloc[:length]

        # Convert 'Bytes' column to numeric, setting non-numeric values to NaN
        df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')
        df['Bytes'] = df['Bytes'].fillna(0)  # Fill NaN values with 0

        # Factorize categorical features to integer labels
        for col in attributes:
            df[col], _ = pd.factorize(df[col])

        x = df.drop(columns=['Label'])
        y = df['Label']

        #debug print
        print(str(len(df)) + " examples in dataset")

        return x, y

    #**********************************************************
    def train_model(self, x, y):
    #
    #   fit wrapper function for all models (same impl for DT and SVM)
    #
    #   params:
    #       x: data
    #       y: labels
    #   returns:
    #       model: trained model
    #**********************************************************
        return self.model.fit(x, y)
    
    #*******************************************************
    def test_model(self, x):
    #
    #   function to test the svm model
    #   
    #   params:
    #       x: test data
    #   returns:
    #       predict: return from SVC predict function
    #*********************************************************
        return self.model.predict(x)

    #*********************************************************
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
    #*********************************************************  
        # Load and preprocess the data
        feature_names = x_train.columns.tolist()
        class_names = sorted(y_train.unique().astype(str))

        # Create and render the decision tree
        dot_data = tree.export_graphviz(model, out_file=None, feature_names=feature_names,
                                    class_names=class_names, filled=True)
        graph = graphviz.Source(dot_data, format="png")
        graph.render("DT/" + self.model_name + "/_dt", view=True)
