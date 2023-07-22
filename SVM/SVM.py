#SVM.py

"""
    author: em
    project: group 7
    class: CS-534 WPI
    last update: July 5, 2023

    Class file for the SVM implementation

    API reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
"""
import pandas as pd
from sklearn.svm import SVC


class SVM:
    """
    SVM Class that contains methods to train, test and render a model.
    """

    # **********************************************************
    def __init__(self, attributes, k_type, d=3, c=0.0, v=True,
                 cache=8000, model_name: str = 'default'
                 ):
        """
        Constructor to initialize the SVM class

        :param attributes: list of column headers to use for training and testing (may not handle full column dimension)
        :type attributes: str[]
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
        self.model = SVC(kernel=k_type, degree=d, C=c,
                         verbose=v, cache_size=cache)            # additional params to init SVC

# **********************************************************
    def set_model(self, trained_model):
        """
        After training the model, update the model object to the trained model

        :param trained_model: model after training
        :type trained_model: sklearn.svm.SVC
        """
        self.model = trained_model

# **********************************************************
    def get_model_name(self):
        """
        Function to return the model name

        :return: name of model in training
        :rtype: str
        """
        return self.model_name

# **********************************************************
    def train_model(self, x, y):
        """
        Fit wrapper function for all models

        :param x: data
        :type x: numpy array or pandas DataFrame
        :param y: labels
        :type y: numpy array or pandas Series
        :return: trained model
        :rtype: sklearn.svm.SVC
        """
        return self.model.fit(x, y)
    
# *****************************************************
    def prepare_data(self, df, attributes, length=-1):
        """
        Function to do specific data preparation for svm. Set x for subset of columns and drop samples from x and y
        past given length value

        :param df: DataFrame to prepare for SVM
        :type df: pandas DataFrame
        :param attributes: columns from loaded dataset (samples)
        :type attributes: list
        :param length: length of dataset to work with (useful for reducing for some models, debugging)
        :type length: int
        :return: prepared data and labels
        :rtype: tuple (pandas DataFrame, pandas Series)
        """
        # Convert 'Bytes' column to numeric, setting non-numeric values to NaN
        df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')
        df['Bytes'] = df['Bytes'].fillna(0)  # Fill NaN values with 0

        # Factorize categorical features to integer labels
        for col in attributes:
            df[col], _ = pd.factorize(df[col])

        # cut examples past length, if no length given don't drop any
        if length == -1:
            length = len(df)
        df = df.drop(df.index[length:])  # train smaller

        # debug print - show new size
        print(str(len(df)) + " examples in dataset")

        # reduce attributes to only those initialized on
        # TODO use the correlation coefficients to determine the attributes
        df = df[self.attributes]

        x = df.drop(columns=['Label'])
        y = df['Label']

        return x, y
    
    def test_model(self, x):
        """
        Test SVM model

        :param x: Test data
        :return: Return from SVC predict function
        """
        return self.model.predict(x)

    # *********************************************************
    def render_model(self, trained_model, x, y):
        """
        Visual representation of the model: for svm, this is a scatter plot of datapoints

        This method is not yet implemented.
        """
    #   TODO: implement - working on either 3D plot or will 
    #       need to adjust the code to plot multiple 2D Linear
    #       SVCs
    # *********************************************************
        pass
