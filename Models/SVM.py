from abc import ABC

from Models.Base import AbstractModel
from sklearn.svm import SVC


class SVM(AbstractModel, ABC):
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
                         verbose=v, cache_size=cache)  # additional params to init SVC
