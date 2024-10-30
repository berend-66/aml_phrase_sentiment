import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
  def __init__(self, n_components):
    """Initialize the Gaussian Mixture Model with a specified number of components."""
    self.model = GaussianMixture(n_components=n_components)

  def fit(self, X):
    """Fit the Gaussian Mixture Model to the data."""
    self.model.fit(X)

  def predict(self, X):
    """Predict cluster labels for each data point and return as a pandas DataFrame."""
    labels = self.model.predict(X)
    labels_df = pd.DataFrame(labels, columns=["Cluster Label"])
    return labels_df

  def predict_proba(self, X):
    """Return probabilities for each Gaussian component for each data point as a DataFrame."""
    probabilities = self.model.predict_proba(X)
    probabilities_df = pd.DataFrame(probabilities, columns=[f"Component {i}" for i in range(probabilities.shape[1])])
    return probabilities_df

  def score(self, X):
    """Calculate the log-likelihood of the data under the model."""
    return self.model.score(X)