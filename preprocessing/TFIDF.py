import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class TFIDF:
  def __init__(self):
    self.tfidf_vectorizer = TfidfVectorizer(
      max_features=5000,            # Limit to the top 1000 features
      ngram_range=(1, 2),           # Include unigrams and bigrams
      min_df=1,                     # Ignore terms that appear in fewer than 3 documents
      max_df=0.7,                   # Ignore terms that appear in more than 90% of the documents
      stop_words='english',         # Use English stopwords
      use_idf=True                  # Use inverse document frequency
    )

  def find_optimal_params(self, X, y):
    # Define the pipeline with TF-IDF and logistic regression
    pipeline = Pipeline([
      ('tfidf', TfidfVectorizer()),
      ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Define the parameter grid for TF-IDF
    param_grid = {
      'tfidf__max_df': [0.7, 0.85, 1.0],  # Test different max_df values
      'tfidf__min_df': [1, 2, 5],         # Test different min_df values
      'tfidf__max_features': [1000, 2500, 5000],  # Different vocab sizes
      'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # Test unigrams, bigrams, trigrams
      'clf__class_weight': ['balanced']  # Adjust for class imbalance 
    }

    # Initialize GridSearchCV with cross-validation (so making validation sets)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    # Fit the grid search to find the best parameters ON LABELED set
    grid_search.fit(X, y)
    # Output the best parameters and the best score
    print("Best TF-IDF Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

  def fit(self, data):
    # Initialize the TF-IDF vectorizer with tunable parameters ---> Tune on validation set using grid search
    tfidf_matrix = self.tfidf_vectorizer.fit_transform(data)

    return pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())

  def transform(self, data):
    tfidf_matrix = self.tfidf_vectorizer.transform(data)
    return pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())