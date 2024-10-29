import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PrincipalComponentAnalysis:
  '''
  Method used for dimensionality reduction. Remove features while preserving
  as much variance as possible. First principal component captures the most variance,
  second PC captures second most, etc.

  Principal components are new variables that are linear combinations of the original
  variables. For each component, the amount of information captured is maximized. These variables
  don't have meaning and are less interpretable.

  1. Standardize the range of continuous initial variables
  2. Compute the covariance matrix to identify correlations
  3. Compute the eigenvectors and eigenvalues of the covariance
      matrix to identify the principal components
  4. Create a feature vector to decide which principal components to keep
  5. Recast the data along the principal components axes
  '''
  def __init__(self, n_components):  # n_components: number of features to reduce to
    self.n_components = n_components
    self.scaler = StandardScaler()  # Instantiate StandardScaler
    self.pca = PCA(n_components=n_components)

  def fit_transform(self, X):
    """Standardizes the data and applies PCA transformation."""
    X_standardized = self.scaler.fit_transform(X)  # Standardize to avoid large vars dominating small vars
    X_pca = self.pca.fit_transform(X_standardized)  # Apply dimensionality reduction
    return X_pca

  def explained_variance(self):
    """Returns the explained variance ratio of the PCA components."""
    return self.pca.explained_variance_ratio_

  def elbow_graph(self):
    """Plots the explained variance ratio for each principal component."""
    explained_variance_ratio = self.explained_variance()
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title('Elbow Graph')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid()
    plt.show()  # Show the plot

  def fit(self, X):
    """Fits the PCA model to the data and transforms it."""
    X_pca = self.fit_transform(X)
    return pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(self.n_components)])