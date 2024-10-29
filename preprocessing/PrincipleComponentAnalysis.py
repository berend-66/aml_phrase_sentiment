import numpy as np
from sklearn.prepocessing import StandardScaler
from sklearn.decomposition import PCA

class PrincipleComponentAnalysis:
  '''
  Method used for dimensionality reduction. Remove features while preserving
  as much variance as possible. First principle component captures the most variance,
  second pc captures second most, etc.

  Principle components are new variables that are linear combinations of the original
  variables. For each component, the amount of information captured is maximized. These variables
  don't have meaning and are less interpretable.

  1. Standardize the range of continuous initial variables
  2. Compute the covariance matrix to identify correlations
  3. Compute the eigenvectors and eigenvalues of the covariance
  matrix to identify the principal components
  4. Create a feature vector to decide which principal components to keep
  5. Recast the data along the principal components axes
  '''
  def __init__(self, n_components): # n_components: number of features to reduce to
    self.n_components = n_components
    self.scaler = StandardScaler
    self.pca = PCA(n_components=n_components)

   def fit_transform(self, X, y):
        X_standardized = self.scaler.fit_transform(X) # standardize to avoid large vars dominating small vars
        X_combined = pd.concat([pd.DataFrame(X_standardized), pd.DataFrame(y, columns=['Outcome'])], axis=1)
        X_pca = self.pca.fit_transform(X_combined) # apply dimensionality reduction
        
        return X_pca
    
    def explained_variance(self):
        return self.pca.explained_variance_ratio_

    def fit(self, X, y):
        X_pca = self.fit_transform(X, y)
        return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(self.n_components)])
