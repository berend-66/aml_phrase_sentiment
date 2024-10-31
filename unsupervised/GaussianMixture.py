import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

class GaussianMixture:
  def __init__(self, **kwargs):
    """Initialize the Gaussian Mixture model with optional scikit-learn parameters."""
    self.model = SklearnGaussianMixture(**kwargs)

  def fit(self, X):
    """Fit the Gaussian Mixture model to the data."""
    self.model.fit(X)

  def predict(self, X):
    """Predict cluster labels for each data point and return as a pandas DataFrame."""
    labels = self.model.predict(X)
    labels_df = pd.DataFrame(labels)
    return labels_df

  def predict_proba(self, X):
    """Return probabilities for each cluster for each data point as a pandas DataFrame."""
    probabilities = self.model.predict_proba(X)
    cluster_names = [f"Cluster {i}" for i in range(probabilities.shape[1])]
    probabilities_df = pd.DataFrame(probabilities, columns=cluster_names)
    return probabilities_df

  def score(self, X):
    """Calculate the log-likelihood of the model on the given data."""
    return self.model.score(X)