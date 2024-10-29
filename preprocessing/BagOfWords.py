import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class BagOfWords:
  def __init__(self, combined_data):
    self.combined_data = combined_data

  def bag_of_words(self, data, threshold_m=1):
    vectorizer = CountVectorizer(binary=True, max_features=threshold_m)
    vectorizer.fit(self.combined_data)
    X = vectorizer.transform(data)

    featurized_data = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names_out())

    return featurized_data

