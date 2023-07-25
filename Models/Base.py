from abc import ABC, abstractmethod
import pandas as pd


class AbstractModel:
    @abstractmethod
    def __init__(self):
        self.model_name = None
        self.model = None

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
        :return: Return from SVC predicts function
        """

        return self.model.predict(x)

    @abstractmethod
    def render_model(self, trained_model, x, y):
        pass
