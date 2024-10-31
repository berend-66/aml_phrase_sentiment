# import packages
import os
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from joblib import dump, load

# import classes
from preprocessing.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from preprocessing.PreProcess import PreProcess
from preprocessing.BagOfWords import BagOfWords
from preprocessing.TFIDF import TFIDF

from unsupervised.GaussianMixture import GaussianMixture

from supervised.LogisticRegression import LogisticRegression
from supervised.KNN import KNN

train_data = pd.read_csv('data/train.csv')
val_data = pd.read_csv('data/val.csv')
test_data = pd.read_csv('data/test.csv')

# Training Data
X_train = train_data['Phrase']
y_train = train_data['Sentiment']

mask = (y_train != -100) # filter out unlabeled data
train_data_clean = train_data[mask]
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# Validation Data
X_val = val_data['Phrase']
y_val = val_data['Sentiment']

# Test Data
X_test = test_data['Phrase']

# --- Preprocessing Data ---
preprocessed_train_path = 'data/X_train_preprocess.csv'
preprocessed_train_clean_path = 'data/X_train_clean_preprocess.csv'
preprocessed_val_path = 'data/X_val_preprocess.csv'
preprocessed_test_path = 'data/X_test_preprocess.csv'

if os.path.exists(preprocessed_train_path): # load data
  X_train_preprocess = pd.read_csv(preprocessed_train_path)['Phrase'].fillna('')
  X_train_clean_preprocess = pd.read_csv(preprocessed_train_clean_path)['Phrase'].fillna('')
  X_val_preprocess = pd.read_csv(preprocessed_val_path)['Phrase'].fillna('')
  X_test_preprocess = pd.read_csv(preprocessed_test_path)['Phrase'].fillna('')
  print("** Preprocessed Data Loaded from Files **")
else:
  pre_processor = PreProcess()
  X_train_preprocess = X_train.apply(pre_processor.process)
  X_train_clean_preprocess = X_train_clean.apply(pre_processor.process)
  X_val_preprocess = X_val.apply(pre_processor.process)
  X_test_preprocess = X_test.apply(pre_processor.process)

  # Save preprocessed data
  X_train_preprocess.to_frame().to_csv(preprocessed_train_path, index=False)
  X_train_clean_preprocess.to_frame().to_csv(preprocessed_train_clean_path, index=False)
  X_val_preprocess.to_frame().to_csv(preprocessed_val_path, index=False)
  X_test_preprocess.to_frame().to_csv(preprocessed_test_path, index=False)
  print("** Data Preprocessed and Saved to Files **")

combined_data = pd.concat([X_train_preprocess, X_val_preprocess, X_test_preprocess])
bag = BagOfWords(combined_data) # Bag of Words

n = 1000
X_train_bag = bag.bag_of_words(X_train_preprocess, threshold_m=n)
X_train_clean_bag = bag.bag_of_words(X_train_clean_preprocess, threshold_m=n)
X_val_bag = bag.bag_of_words(X_val_preprocess, threshold_m=n)
X_test_bag = bag.bag_of_words(X_test_preprocess, threshold_m=n)
print("** Bag of Words Completed **")
tfidf = TFIDF() # TF-IDF
X_train_clean_tfidf = tfidf.fit(X_train_clean_preprocess)
X_train_tfidf = tfidf.fit(X_train_preprocess)
print("** TF-IDF Completed **")

# pca = PrincipalComponentAnalysis(200) # Principle Component Analysis
# X_train_bag_pca = pca.fit(X_train_bag)
# pca.elbow_graph()
# X_train_clean_bag_pca = pca.fit(X_train_clean_bag)
# X_train_tfidf_pca = pca.fit(X_train_tfidf)
# X_train_clean_tfidf_pca = pca.fit(X_train_clean_tfidf)

# --- Unsupervised Learning ---
gmm = GaussianMixture(n_components=5, random_state=42) # Gaussian Mixture

### TFIDF -> GMM ###
gmm_tfidf_model_path = 'models/gmm_tfidf_model.joblib'
if os.path.exists(gmm_tfidf_model_path): # load model
  gmm = load(gmm_tfidf_model_path)
  print("** GMM model loaded from file **")
else: # train model
  gmm.fit(X_train_tfidf)
  dump(gmm, gmm_tfidf_model_path)
  print("** GMM model trained and saved to file **")

labels = gmm.predict(X_train_tfidf).values.flatten()

unlabeled_indices = np.where(y_train == -100)[0] # Indices of unlabeled data
y_train_tfidf_gmm = y_train.copy()
y_train_tfidf_gmm[unlabeled_indices] = labels[unlabeled_indices]

### TFIDF -> GMM ###

print("** GMM Completed **")

# --- Supervised Learning --- #

# logr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # Logistic Regression
# ### TFIDF -> Logistic Regression ###
# logr.fit(X_train_clean_tfidf, y_train_clean)
# predictions = logr.predict(X_train_clean_tfidf)
# print("LOGISTIC REGRESSION (TF-IDF)")
# print(f1_score(y_train_clean, predictions, average='weighted'))
# ### Bag of Words -> Logistic Regression ###
# logr.fit(X_train_clean_bag, y_train_clean)
# predictions = logr.predict(X_train_clean_bag)
# print("LOGISTIC REGRESSION (Bag of Words)")
# print(f1_score(y_train_clean, predictions, average='weighted'))
# ### TFIDF -> GMM -> Logistic Regression ###
# pass
# ### Bag of Words -> GMM -> Logistic Regression ###
# pass
# ### TFIDF -> KMEANS -> Logistic Regression ###
# pass
# ### Bag of Words -> KMEANS -> Logistic Regression ###
# pass
print("** Logistic Regression Completed **")

knn = KNN(n_neighbors=3) # KNN

### TFIDF -> KNN ###
knn_tfidf_model_path = 'models/knn_tfidf_model'
if os.path.exists(knn_tfidf_model_path): # load model
  knn = load(knn_tfidf_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_clean_tfidf, y_train_clean)
  dump(knn, knn_tfidf_model_path)
  print("** KNN model trained and saved to file **")
predictions = knn.predict(X_train_clean_tfidf)
print("KNN (TF-IDF)")
print(f1_score(y_train_clean, predictions, average='weighted'))

### Bag of Words -> KNN ###
knn_bag_model_path = 'models/knn_bag_model'
if os.path.exists(knn_bag_model_path): # load model
  knn = load(knn_bag_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_clean_bag, y_train_clean)
  dump(knn, knn_bag_model_path)
  print("** KNN model trained and saved to file **")
predictions = knn.predict(X_train_clean_bag)
print("KNN (TF-IDF)")
print(f1_score(y_train_clean, predictions, average='weighted'))

print("** KNN Completed **")
