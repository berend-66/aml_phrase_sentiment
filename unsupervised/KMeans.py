import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans

class KMeans:
  def __init__(self, n_clusters=8, random_state=None, **kwargs):
    """
    Initialize the KMeans classifier with optional parameters.
    Parameters:
    - n_clusters (int): Number of clusters to form.
    - random_state (int): Seed for random number generation.
    - **kwargs: Additional parameters for sklearn's KMeans.
    """
    self.model = SklearnKMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)

  def fit(self, X):
    """
    Fit the KMeans model to the data.
    Parameters:
    - X (array-like or DataFrame): Training data.
    """
    self.model.fit(X)

  def predict(self, X):
    """
    Predict the closest cluster each sample in X belongs to.
    Parameters:
    - X (array-like or DataFrame): Data for which to predict cluster labels.
    Returns:
    - labels (Series): Cluster labels for each data point.
    """
    labels = self.model.predict(X)
    return pd.Series(labels, name="Predicted Cluster")

  def fit_predict(self, X):
    """
    Compute cluster centers and predict cluster index for each sample.
    Parameters:
    - X (array-like or DataFrame): Training data.
    Returns:
    - labels (Series): Cluster labels for each data point.
    """
    labels = self.model.fit_predict(X)
    return pd.Series(labels, name="Predicted Cluster")

  def get_cluster_centers(self):
    """
    Get the coordinates of cluster centers.
    Returns:
    - centers (DataFrame): Cluster centers as a DataFrame.
    """
    centers = self.model.cluster_centers_
    return pd.DataFrame(centers, columns=[f"Feature {i}" for i in range(centers.shape[1])])

  def inertia(self):
    """
    Calculate the sum of squared distances of samples to their closest cluster center.
    Returns:
    - inertia (float): Sum of squared distances of samples to closest cluster center.
    """
    return self.model.inertia_

  def score(self, X):
    """
    Calculate the opposite of the inertia on the given data.
    Parameters:
    - X (array-like or DataFrame): Data to score.
    Returns:
    - score (float): Negative of the inertia (useful for model evaluation).
    """
    return -self.model.score(X)