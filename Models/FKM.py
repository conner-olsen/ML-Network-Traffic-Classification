# FKM.py

"""
    author: em, co
    project: group 7 term project
    class: CS-534 Artificial Intelligence WPI
    date: July 15, 2023,
    last update: July 25, 2023

    class to implement the fuzzy k-modes model

    API ref:
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.html#fuzzy-compare
    https://joblib.readthedocs.io/en/latest/parallel.html
"""
import math
import statistics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Util.Util import get_results_location


class FKM:
    """
    FKM class to implement the fuzzy k-modes model
    """

    # logging globals for testing
    DEBUG_PRINT = True  # set to true for testing
    DEBUG_NOPRINT = False  # always skip printing these but can remove "no" to expose

    # ********************************************************
    def __init__(self, k=8, model_name: str = "default"):
        """
        FKM class constructor - initialize model, set k
        :param k: number of clusters
        (default is 8) value greater than 8 may cause issues with math on giant matrices
        :param model_name: name the FM models
        """
        self.k = k  # number of clusters
        self.model_name = model_name  # name of the model (user named)
        self.model_type = "FKM"  # type of model (FKM)
        self.centroids = []  # selected at random (Lloyd's)
        self.means = []  # mean of points in each cluster (maybe use for p)
        self.cluster_variance = []  #
        self.p = []  # probability metric for determining group membership
        self.clusters = dict()
        self.accuracy_percent = None
        self.model = {
            "name_": "FMK",
            "k_": self.k,
            "clusters_": self.clusters,
            "centroids_": self.centroids,
            "p_": self.p,
            "score": [],
            "samples_": None,
        }  # fill vals after training, testing, etc

    # ********************************************************
    def get_k(self):
        """
        function to get the number of clusters
        :return: numbers of clusters
        """
        return self.k

    # ********************************************************
    def prepare_data(self, df, num_records: int = 300000):
        """
        Implements model-specific data manipulations.
        It changes Panda's DataFrame to NumPy arrays.

        This function also prepares random centroids from the entire dataset before splitting,
        then fits the train to centroids, making assumptions about future points,
        testing the hypothesis in test, and evaluating after train and test.

        It also modifies the `self.centroids` and `self.model["centroids_"]` attributes in-place,
        attaching the labels to the centroids, based on dataset indices.

        :param num_records: The number of records to use from the dataset.
                            If None, use the entire dataset.
                            Default is 300000 records to decrease runtime.
        :param df: The dataset loaded from pandas read_csv function.
        :type df: DataFrame

        :return: The modified version of the initial dataset with dropped label column, The labels from the dataset,
                 saved for evaluation and consistency with other APIs.
        :rtype: ndarray, ndarray
        """

        if num_records is not None:
            df = df.head(num_records)

        # Drop labels
        x = df.drop(columns="Label")

        # factorize the Flags column
        x["Flags"] = pd.factorize(x["Flags"])[0]

        # Convert to numpy
        x = x.to_numpy()
        y = df["Label"]
        y = y.values.ravel()

        # Print the count of true labels
        print(f"{np.count_nonzero(y == 1)} true")

        # Remove centroids, centroid indexes
        x = self.random_grouping(x)

        # Remove y_vals, put the label back at the end
        yc = []
        for c in self.centroids:
            yc.append(y[c[0]])
            y = np.delete(y, c[0], 0)
        print(f"centroid targets {yc}")

        # Insert labels back to the centroids
        self.centroids = np.insert(self.centroids, len(self.centroids[0]), yc, axis=1)

        # Update the model centroids
        self.model["centroids_"] = self.centroids

        return x, y

    # *************************************************************
    def train_model(self, x_train, y_train):
        """
        Trains the FKM model, performing clustering on the input data.

        The fuzzy c-means (FKM) algorithm is applied, transforming the data
        through fuzzification, finding the mean, and finally defuzzification.

        :param x_train: A nD array representing dataset samples (minus centroids)
        :type x_train: numpy.ndarray
        :param y_train: A placeholder input, it is not used in training.
                        But it is passed in to maintain API consistency.
        :type y_train: numpy.ndarray
        :return: A dictionary representing an FKM model which includes centroids,
                 X (data samples) with membership, k value (the number of clusters),
                 and a probability array.
                 The returned model does not include filled clusters and scores.
        :rtype: dict
        """

        x_mems = self.fuzzify(x_train)
        print(f"{len(x_mems[0])} after fuzzify")
        idx = len(x_mems[0]) - self.k
        print(x_mems[0, idx:])
        self.find_mean(x_mems)
        self.defuzzify(
            x_mems
        )  # crisp data - maybe actually need this at the end of training

        return self.model

    # ****************************************************************
    def test_model(self, x_test):
        """
        Tests the Fuzzy K-Means (FKM) model.

        This function calculates the probability of each data point in `x_test`
        belonging to each cluster, then assigns each data point to the cluster
        with the highest probability.

        :param x_test: the test samples to be evaluated.
        :type x_test: numpy.ndarray
        :return: an array of predicted class labels for each data point in `x_test`.
        :rtype: numpy.ndarray
        """
        # Calculate the membership of each test sample to each centroid
        memberships = np.empty((x_test.shape[0], self.k), dtype=float)
        for i, x in enumerate(x_test):
            for j, centroid in enumerate(self.model["centroids_"]):
                memberships[i, j] = self.calc_dist(
                    x, centroid[2:]
                )  # should start from index 2 assuming that index 1 is the centroid index

        # Normalize membership values such that each row sums to 1
        epsilon = 1e-10  # very small value
        memberships /= np.sum(memberships, axis=1, keepdims=True) + epsilon

        # Assign each sample to the cluster with the highest membership value
        predictions = np.argmax(memberships, axis=1)

        return predictions

    # *********************************************************
    def evaluate(self, y_test, predictions):
        """
        Evaluates how accurately the data was clustered against known labeled data by examining various
        clustering metrics.

        :param predictions: An array representing the predicted labels.
        :param y_test: An array representing the targets.
                       FKM model instance utilizes the cluster indices model['clusters_'].
        :return: None.

        It looks at the following clustering metrics:
            - composition: Compare clustered points to label value.
                           Calculates the number of true
              values as a percent of total cluster points.
            - homogeneity: Are all points in the cluster the same label.
            - completeness: Are all targets isolated into clusters or are they more distributed.
            - v_measure: Harmonic mean of homogeneity and completeness.

        The function will print the results on the console and also write them to a file.
        """
        n_true = np.count_nonzero(y_test == 1)
        n_samples = len(y_test)
        composition, counts_per_cluster = self.get_composition(
            self.model["clusters_"], y_test
        )
        print(f"Composition: {composition}")

        homogeneity = self.get_homogeneity(counts_per_cluster, n_true, n_samples)

        completeness = self.get_completeness(counts_per_cluster, n_true, n_samples)

        v_measure = self.get_v_measure(homogeneity, completeness)

        composition_mean = None
        if composition:
            composition_mean = statistics.mean(composition)

        metrics = {
            "composition_": composition_mean,  # Updated line
            "homogeneity_": homogeneity,
            "completeness_": completeness,
            "v_measure_": v_measure,
            "accuracy_": self.accuracy_percent,
        }

        self.model["score_"] = metrics

        # Write the evaluation metrics to a file
        with open(get_results_location("FKM", self.model_name), "a") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

        # Also print the results to stdout
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    # ******************************************************
    def render_model(self, x, y):
        """
        Plots and saves the XY scatter of clustered datapoints based on the FKM model provided.

        :param x: The input samples.
                  It is a feature array.
        :param y: The target values.
                  Binary target values are assumed.
        :returns: None

        Note:
            As the targets are binary, the cluster index will range from 1 to 2.
            This function is a part of the FKM class and does not work standalone.
            It does not yet support multi-class datasets and needs to be further developed for the same.
        """
        plt.figure(figsize=(10, 8))
        for i, centroid in enumerate(self.model["centroids_"]):
            points = x[
                self.model["clusters_"][i]
            ]  # get points that belong to this cluster
            plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i+1}")

        centroid_coords = [centroid[1:] for centroid in self.model["centroids_"]]
        centroid_coords = np.array(centroid_coords)

        plt.scatter(
            centroid_coords[:, 0],
            centroid_coords[:, 1],
            marker="x",
            s=300,
            linewidths=5,
            color="black",
            label="Centroids",
        )

        plt.title("FKM Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    # **********************************************************
    def get_composition(self, clusters, y):
        """
        Calculate how accurately the clusters are in comparison to the original dataset labels.

        :param clusters: kD array of indices for each datapoint in each cluster.
        :type clusters: array-like

        :param y: Original dataset labels.
        :type y: array-like

        :returns:
            - composition - Ratio of true samples to the respective cluster size.
            - counts - A dictionary where the keys are cluster numbers, and the values are pairs,
              where the first element is the true count, and the second one is the total points in the cluster.

        :rtype: (list, dict)

        """
        composition = []
        counts = dict()
        for i in range(self.k):

            if i in clusters:  # Check key existence before operation
                c = 0
                counts[i] = []
                print(f"{len(clusters[i])} datapoints in cluster {i}")

                for data_idx in clusters[i]:
                    if y[data_idx] == 1:
                        c += 1
                print(f"{c} true!")
                counts[i].append(c)
                counts[i].append(len(clusters[i]))

                if len(clusters[i]) > 0:
                    composition.append(c / len(clusters[i]))
                else:
                    composition.append(0)

        p_true_y = sum(label for label in y if label)

        self.accuracy_percent = p_true_y / len(y)

        print("The composition of the clusters")
        return composition, counts

    # **********************************************************************
    def get_homogeneity(self, counts_per_cluster, n_true, n_samples):
        """
        Calculate the homogeneity score to determine if clusters consist solely of one label value
        (true, false, or multiclass).

        :param counts_per_cluster: 2-D Iterable representing counts associated with each cluster.
        :param n_true: Integer representing the count of 'true' labels.
        :param n_samples: Integer indicating the total number of samples.

        :return: The homogeneity score for the current clustering.
                 A score close to 1 indicates that the clusters are predominantly composed of only one label.

        Homogeneity score is calculated as follows:

        For each cluster, it measures (n_true / n_samples) * log2(counts_per_cluster[c][0] / counts_per_cluster[c][1]).
        The output is the sum of these measurements across all clusters around n_true.

        """
        h = 0
        for c in range(self.k):  # k clusters
            if (
                counts_per_cluster.get(c) and len(counts_per_cluster[c]) > 1
            ):  # Check if key is present and if a list at key has at least 2 elements
                if counts_per_cluster[c][0] > 0 and counts_per_cluster[c][1] > 0:
                    h += (n_true / n_samples) * math.log(
                        counts_per_cluster[c][0] / counts_per_cluster[c][1], 2
                    )
        h /= self.k
        print(str(h) + " homogeneous")
        return h

    # ********************************************************************
    def get_completeness(self, counts_per_cluster, n_true, n_samples):
        """
        Measure how separated the clusters are (completeness)

        :param counts_per_cluster: A dictionary where the key is the cluster index and the values are the
                number of true points and the count of clusters
        :type counts_per_cluster: dict
        :param n_true: the Number of true points
        :type n_true: int
        :param n_samples: Total number of samples
        :type n_samples: int
        :return: Completeness score; a higher value means the clusters are more separated
        :rtype: float

        """
        hck_k = 0
        for c in range(self.k):  # k clusters
            hck = 0
            if counts_per_cluster.get(c) is not None and len(counts_per_cluster[c]) > 1:
                if counts_per_cluster[c][0] > 0 and counts_per_cluster[c][1] > 0:
                    hck = (n_true / n_samples) * math.log(
                        counts_per_cluster[c][0] / counts_per_cluster[c][1], 2
                    )
            hck_k += 1 - (-hck / self.k)
        print(str(hck_k) + " completeness")
        return hck_k / self.k

    # **********************************************************************
    def get_v_measure(self, homogeneity, completeness):
        """
        Compute the harmonic mean of the homogeneity and completeness score,
        often referred to as v-measure or Normalized
        Mutual Information (NMI).

        :param homogeneity: The homogeneity score.
        :type homogeneity: float
        :param completeness: The completeness score.
        :type completeness: float
        :return: The v-measure score.
        :rtype: float
        """
        return 2 * (homogeneity * completeness) / (homogeneity + completeness)

    # *******************************************************
    def bin_search_rm(self, x):
        """
        Speed up removing the centroids from the samples with binary search and remove all centroids
        in the data that match the self.centroids array.
        Indexes are saved for mapping back to the point in validation to avoid dataset shift.

        :param x: nD array, the dataset (samples)
        :return: tuple of an array and list,
                 updated X with centroid points removed and array of centroid indices
        """
        c_idx = []
        for i in range(self.k):
            start = 0
            end = len(x)
            print(self.centroids[i, :])  # debug print
            steps = 0
            while start <= end:
                mid = int((start + end) / 2)
                steps += 1
                if any((x[:mid] == self.centroids[i]).all(1)):
                    end = mid
                else:
                    start = mid

                if (x[mid] == self.centroids[i]).all():
                    print("found in " + str(steps) + " steps at idx " + str(mid))
                    c_idx.append(mid)  # for recenter have to replace
                    x = np.delete(x, mid, 0)
                    break

        print("num points after removing centroids: " + str(len(x)))

        return x, c_idx

    # *************************************************************
    def random_grouping(self, x):
        """
        Separate X into k subsets at random using Lloyd's algorithm - select k centroids
        at random, and classify all other data to correlate to centroid.

        :param x: array, samples to a divide
        :return: array, updated data
        """
        self.centroids = x[np.random.choice(x.shape[0], self.k, replace=False)]
        print(self.centroids)
        print("number of centroids: " + str(len(self.centroids)))

        x, c_idx = self.bin_search_rm(x)

        self.centroids = np.insert(self.centroids, 0, c_idx, axis=1)
        for c in self.centroids:
            print(str(c[0]) + ":" + str(c[1:]))

        return x

    # *************************************************************
    def calc_dist(self, val1, val2):
        """
        Analyze distance between points given their type.

        :param val1: Any value from matrix[r][c]_A
        :param val2: Any value from matrix[r][c]_B
        :return: float, similarity, scaled 0-1, 1 being most similar (100%)
        """
        val_t = type(val1)
        if val_t is int or val_t is float:  # deviation metric
            diff = val1 - val2
            sum_dist = val1 + val2
            if np.all(
                diff == 0
            ):  # You can replace np.all with np.any based on your requirement
                return 1  # identical
            else:
                canberra = 1 - (abs(diff) / sum_dist)
                return canberra
        elif val_t is str:  # jaro (str)
            raise Exception(
                f"Error: jaro_distance expects scalar inputs, but received {val1} and {val2}"
            )
        else:  # binary (no std type)
            if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                return 1 if np.array_equal(val1, val2) else 0
            else:
                return 1 if val1 == val2 else 0

    # *************************************************************
    def fuzzify(self, x):
        """
        Measure the distance of datapoint from centroid
        randomly assign a membership value which will get converted to distance later.

        :param x: array, input data
        :return: array, updated data with membership values
        """
        n_cols = len(x[0])
        print("before fuzz" + str(n_cols))
        member = []
        check_sts = int(len(x) / 10)
        for i in range(len(x)):  # for each datapoint
            if i % check_sts == 0:
                print("status {t: .2f}%".format(t=(i / len(x) * 100)))
            tmp = []
            for j in range(len(self.centroids)):  # compare point to centroid
                p_mem = sum(
                    self.calc_dist(self.centroids[j][k + 1], x[i][k])
                    for k in range(self.k)
                )
                tmp.append(p_mem / n_cols)  # average similarity over all features
            member.append(tmp)  # add row of cluster membership vals per point

        x = np.concatenate([x, member], axis=1)

        if self.DEBUG_PRINT:
            print(f"num cols:{len(x[0])}")
            print(f"num rows: {len(x)}")
            print(x[0])

        self.model["samples_"] = x
        return x

    # *************************************************************
    def defuzzify(self, x):
        """
        Crisps up our datapoints in the end to validate.
        At each point, finds which of the clusters it most
        fits into and adds it to the dictionary for future plots.

        Args:
            x : numpy.ndarray
                2D array containing samples and their membership values.

        Returns:
            None
        """
        row_sz = len(x[0])  # Total rows
        print(f"{str(row_sz)} rows")

        mem = row_sz - self.k  # index where membership values starts in X
        keys = np.arange(self.k)

        self.clusters = {k: [] for k in keys}

        for r in range(len(x)):  # each row gets max of last k cols
            tmp = x[r, mem:]
            maxima = max(tmp)
            idx = np.where(maxima == tmp)[0]  # cluster col with max val
            c = idx[0]
            self.clusters[c].append(r)  # save sample row idx

        group_szs = [{i: len(self.clusters[i])} for i in range(self.k)]
        self.model["clusters"] = self.clusters

        sample_sz = len(x)
        self.p = self.find_probabilities(sample_sz, group_szs)

    def find_mean(self, x):
        """
        Computes the average weight of any given sample in
        a cluster (how well they match each other)

        Args:
            x : numpy.ndarray
                2D array containing samples and their membership values.

        Returns:
            None
        """
        idx = len(x[0]) - self.k  # index where membership value starts
        sum_arr = np.sum((x[:, idx:]), axis=0)
        self.means = [sum_arr[i] / len(x) for i in range(len(sum_arr))]

        self.cluster_variance = [
            statistics.variance(x[:, idx], self.means[i]) for i in range(self.k)
        ]

    def find_probabilities(self, sample_sz, cluster_sizes):
        """
        Computes predictive probabilities for future samples
        to become members of clusters.

        Args:
            sample_sz : int
                Total number of samples.
            cluster_sizes : list
                Sizes of the clusters.

        Returns:
            list : Predictive probability distribution over the clusters.
        """
        probs = [list(s.values())[0] / sample_sz for s in cluster_sizes]
        return probs
