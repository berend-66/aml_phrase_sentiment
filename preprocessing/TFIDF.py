import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
  def fit(self, data):
    # Initialize the TF-IDF vectorizer with tunable parameters ---> Tune on validation set using grid search
    tfidf_vectorizer = TfidfVectorizer(
      max_features=1000,            # Limit to the top 1000 features
      ngram_range=(1, 2),           # Include unigrams and bigrams
      min_df=3,                     # Ignore terms that appear in fewer than 3 documents
      max_df=0.9,                   # Ignore terms that appear in more than 90% of the documents
      stop_words='english',         # Use English stopwords
      sublinear_tf=True,            # Apply sublinear scaling to term frequency
      use_idf=True                  # Use inverse document frequency
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    return pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())