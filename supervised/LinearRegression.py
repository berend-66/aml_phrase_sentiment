import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

class LinearRegression:
    def __init__(self):
        """Initialize the Linear Regression model."""
        self.model = SklearnLinearRegression()

    def fit(self, X, y):
        """Fit the linear regression model to the data."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions using the linear regression model."""
        predictions = self.model.predict(X)
        predictions_df = pd.DataFrame(predictions, columns=["Predictions"]).round(0)
        return predictions_df

    def score(self, X, y):
        """Calculate R-squared score."""
        return self.model.score(X, y)