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

  def find_optimal_params(self, X):
    param_grid = {
      'n_components': [5],  # numbers of clusters
      'init_params': ['kmeans', 'random'],  # Initialization methods
      'tol': [1e-3, 1e-4],  # Convergence thresholds
      'max_iter': [50, 100]  # Maximum iterations
    }

    best_bic = np.inf
    best_params = None
    best_gmm = None

    for n_components in param_grid['n_components']:
      for init_params in param_grid['init_params']:
        for tol in param_grid['tol']:
          for max_iter in param_grid['max_iter']:
            # Initialize and fit GMM with the current set of parameters
            gmm = GaussianMixture(
              n_components=n_components,
              init_params=init_params,
              tol=tol,
              max_iter=max_iter,
              random_state=42
            )
            gmm.fit(X)
            # Calculate BIC
            bic = gmm.bic(X)
            print(f"n_components={n_components}, init_params={init_params}, tol={tol}, max_iter={max_iter} -> BIC={bic:.2f}")
            
            # Update best parameters if this BIC is lower
            if bic < best_bic:
              best_bic = bic
              best_params = {
                'n_components': n_components,
                'init_params': init_params,
                'tol': tol,
                'max_iter': max_iter
              }
              best_gmm = gmm

    return best_params, best_bic, best_gmm