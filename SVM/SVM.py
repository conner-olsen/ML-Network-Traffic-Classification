#SVM.py

'''
    author: em
    project: group 7 
    class: CS-534 WPI
    last update: July 5, 2023

    Class file for the SVM implementation

    API reference: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
'''
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

class SVM:

    #**********************************************************
    def __init__(self, attributes, k_type, d=3, c=0.0, v=True,
                 cache = 8000, model_name: str='default' 
                 ):
    #
    #   constructor
    #   
    #   params:
    #       attributes (str[]): list of column headers to use
    #           for training and testing (may not handle full 
    #           column dimension)
    #       k_type (str): kernel type - poly,rbf, linear, etc
    #       d (int): dimension (for poly kernel)
    #       c (float): regularization parameter
    #       v (bool): verbose (good for debugging)
    #       cache (int): cache size (bytes)
    #       model_name (str): name of the SVM being trained
    #**********************************************************
        self.model_name = model_name
        self.attributes = attributes
        self.model = SVC(kernel=k_type,degree=d, C=c, 
                         verbose=v, cache_size=cache)            #additional params to init SVC

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

    #**********************************************************
    def train_model(self, x, y):
    #
    #   fit wrapper function for all models
    #
    #   params:
    #       x: data
    #       y: labels
    #   returns:
    #       model: trained model
    #**********************************************************
        return self.model.fit(x, y)
    
    #*****************************************************
    def prepare_data(self, df, attributes, length=-1):
    #
    #   function to do specific data preparation for svm
    #       set x for subset of columns
    #       drop samples from x and y past given length value
    #
    #   params:
    #       x: columns from loaded dataset (samples)
    #       y: labels from dataset (targets)
    #       length: length of dataset to work with (useful for 
    #           reducing for some models, debugging)
    #   TODO: check that updated data files works with this
    #       Conner, Corey mentioned changes since I tested these
    #*****************************************************
        # Convert 'Bytes' column to numeric, setting non-numeric values to NaN
        df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')
        df['Bytes'] = df['Bytes'].fillna(0)  # Fill NaN values with 0

        # Factorize categorical features to integer labels
        for col in attributes:
            df[col], _ = pd.factorize(df[col])

        #cut examples past length, if no length given don't drop any
        if (length == -1):
            length = len(df)
        df = df.drop(df.index[length:] ) #train smaller

        #debug print - show new size
        print(str(len(df)) + " examples in dataset")

        #reduce attributes to only those initialized on
        #TODO use the correlation coefficients to determine the attributes
        df = df[self.attributes]

        x = df.drop(columns=['Label'])
        y = df['Label']

        return x, y
    
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
    def render_model(self, trained_model, x, y):
    #
    #   visual representation of the model: for svm, this is 
    #       a scatter plot of datapoints
    #
    #   TODO: implement - working on either 3D plot or will 
    #       need to adjust the code to plot multiple 2D Linear
    #       SVCs
    #*********************************************************  
        pass
