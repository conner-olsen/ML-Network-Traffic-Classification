"""
SVM Model
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from Models.Base import AbstractModel


class SVM(AbstractModel):
    def __init__(
        self,
        k_type="rbf",
        d=3,
        c=1.0,
        n_jobs=-1,
        v=True,
        cache=8000,
        model_name: str = "default",
    ):
        """
        Constructor to initialize the SVM class

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
        super().__init__()
        self.model_name = model_name
        self.attributes = ["Dst_Pt", "Src_IP", "Bytes", "Label"]
        svm = SVC(kernel=k_type, degree=d, C=c, verbose=v, cache_size=cache)
        self.model = BaggingClassifier(svm, max_samples=0.5, n_jobs=n_jobs)
        self.model_type = "SVM"

    def prepare_data(self, df, num_records=300000):
        """
        Prepare data for training.
        This includes:
            - Selecting only the attributes we want to use.
            - Selecting only the first num_records records.

        :param num_records: Number of records to use from the dataset.
                            Default is 50,000 for SVM as it is very slow.
        :param df: "Data frame" Data loaded from csv.
        :type df: pandas.DataFrame
        :return: tuple of feature matrix and label vector.
        :rtype: tuple[pandas.DataFrame, pandas.Series]
        """
        df = df[self.attributes]

        if num_records is not None:
            df = df.head(num_records)

        x = df.drop(columns=["Label"])
        y = df["Label"]

        print(f"{len(df)} examples in dataset")

        return x, y

    def render_model(self, x, y):
        """
        Visual representation of the model using plotly for an interactive 3D visualization

        :param x: The training data
        :type x: DataFrame
        :param y: The training labels
        :type y: Series
        :returns: Nothing, display the rendered model interactively
        """

        x_lim = [x["Dst_Pt"].min(), x["Dst_Pt"].max()]
        y_lim = [x["Src_IP"].min(), x["Src_IP"].max()]
        z_lim = [x["Bytes"].min(), x["Bytes"].max()]

        # create grid to evaluate model
        xx, yy, zz = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], 20),
            np.linspace(y_lim[0], y_lim[1], 20),
            np.linspace(z_lim[0], z_lim[1], 20),
        )

        xy = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        xy_df = pd.DataFrame(xy, columns=x.columns)
        decision_values = self.model.decision_function(xy_df)
        decision_values = decision_values.reshape(xx.shape)

        fig = go.Figure()

        # Plot decision boundary
        fig.add_trace(
            go.Isosurface(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                value=decision_values.flatten(),
                isomin=-0.5,
                isomax=0.5,
                opacity=0.6,
                surface=dict(count=3),
                colorscale="RdBu",
                cmin=-1,
                cmax=1,
            )
        )

        # Add data points
        fig.add_trace(
            go.Scatter3d(
                x=x["Dst_Pt"],
                y=x["Src_IP"],
                z=x["Bytes"],
                mode="markers",
                marker=dict(color=y, colorscale="RdBu", size=5, opacity=0.8),
            )
        )

        fig.show()
