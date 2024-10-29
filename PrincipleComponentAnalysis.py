import numpy as np
from sklearn.prepocessing import StandardScaler
from sklearn.decomposition import PCA

class PrincipleComponentAnalysis:
  '''
  Method used for dimensionality reduction. Remove features while preserving
  as much variance as possible. First principle component captures the most variance,
  second pc captures second most, etc.
  '''
  def __init__(self, n_components): # n_components: number of features to reduce to
    self.n_components = n_components
    self.scaler = StandardScaler
    self.pca = PCA(n_components=n_components)

   def fit_transform(self, X, y):
        X_standardized = self.scaler.fit_transform(X)
        X_combined = pd.concat([pd.DataFrame(X_standardized), pd.DataFrame(y, columns=['Outcome'])], axis=1)
        X_pca = self.pca.fit_transform(X_combined)
        
        return X_pca
    
    def explained_variance(self):
        return self.pca.explained_variance_ratio_

    def fit(self, X, y):
        X_pca = self.fit_transform(X, y)
        return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(self.n_components)])
