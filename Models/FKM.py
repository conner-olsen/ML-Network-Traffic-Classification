#KM.py

'''
    author: EM
    project: Group 7
    class: CS534 Artificial Intelligence WPI
    date: July 23, 2023

    alternative class to FKM - FKM is very complicated and may
    be more of a long-term project. K-means has a sklearn library
    already, while FKM is being written from scratch. Decision to impl.
    this to fulfill project requirements, but hoping to continue work with
    FKM for future research project paper, not necessarily on same dataset.

    API ref:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict
'''

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, \
    rand_score, \
    silhouette_score, \
    adjusted_mutual_info_score, \
    completeness_score

from Models.Base import AbstractModel
from Util.Util import get_results_location


class FKM(AbstractModel):
    VERBOSE = 0     # global verbose (passed to model so we have status msg in dev.)

    # *********************************************************************
    def __init__(self, k=8, model_name='default'):
        ##
        #   @brief: K-Means class constructor - initialize model, set k
        #
        #   @params:
        #       k (int): number of clusters (default is 8)
        #       model_name (str): name the FM models
        # *********************************************************************
        self.k = k
        self.model_name = model_name
        self.model = KMeans(n_clusters = self.k, algorithm='lloyd', random_state=0,
                            verbose=self.VERBOSE)

    # ***************************************************************************
    def prepare_data(self, df):
        #
        #   @brief: additional data processing needed for specific model, pass both
        #       train and test df's to split into samples and targets
        #
        #   @params:
        #       df (dataframe): the whole, pre-processed dataframe (samples and target)
        #   @returns:
        #       X (dataframe): the samples (train or test)
        #       y (dataframe): the targets (train or test)
        # ****************************************************************************
        X = df.drop(columns=['Label'])
        y = df['Label']

        return X, y

    # ***************************************************************************
    def train_model(self, x_train, y_train):
        ##
        #   @brief: train the model - fit to X train samples
        #
        #   @params:
        #       X_train (dataframe): the training dataset samples
        #       y_train (dataframe): None, retained for API consistency
        #
        #   @returns:
        #       trained model
        # ***************************************************************************
        self.model = self.model.fit(x_train)
        return self.model

    # ****************************************************************************
    def test_model(self, x_test):
        ##
        #   @brief: test the model - the "predict" function on fitted model and test
        #       data
        #
        #   @params:
        #       X_test (dataframe): the testing dataset samples
        #   @returns:
        #       predictions from "predict" function call
        # ****************************************************************************
        return self.model.predict(x_test)

    # ****************************************************************************
    def evaluate(self, x, y, predict):
        ##
        #   @brief: Evaluate clustering accuracy
        #
        #   @params:
        #       model (KMeans model): returned from fit() or predict()
        #       y (dataframe): targets from training set or testing set
        #   @returns:
        #       nothing, prints to result file
        #
        #   API ref:
        #   https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        # ****************************************************************************
        # Compute evaluation metrics
        metrics = {
            'Random Index': rand_score(y, predict),
            'Adjusted Random Index': adjusted_rand_score(y, predict),
            'Silhouette': silhouette_score(x, predict),     # taks very long on lg data
            'Adjusted Mutual Info': adjusted_mutual_info_score(y, predict),
            'Completeness': completeness_score(y, predict)
        }

        # Write the evaluation metrics to a file
        with open(get_results_location("KM", self.model_name), 'a') as f:
            for metric, value in metrics.items():
                f.write(f'{metric}: {value}\n')

        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f'{metric}: {value}')

    # ****************************************************************************
    def render_model(self, model, x, y):
        ##
        #   @brief: visualize the k clusters
        #
        #   @params:
        #       model: fitted model
        #       X (dataframe): dataset samples
        #       y (dataframe): dataset targets (unused, kept for API consisitency)
        #
        #   @returns:
        #       nothing, but displays and saves a .png file of cluster plot
        # *****************************************************************************
        # print(model.labels_)       #train labels, predict is on test

        og = x[model == 0]  # indexes

        # this does not really represent the relation of the clusters well
        plt.scatter(og.iloc[:, 0], og.iloc[:, 1], color="black")
        # print(og)
        plt.scatter(model, y, color='red')
        # plt.scatter(og, y, color='red')
        # plt.scatter(self.model.cluster_centers_[0],
        #    self.model.cluster_centers_[3],
        #    color="blue", marker="*")
        # plt.scatter(self.model.cluster_centers_[2],
        #    self.model.cluster_centers_[0],
        #    color="red", marker="*")

        plt.show()


