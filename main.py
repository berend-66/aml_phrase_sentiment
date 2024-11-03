# import packages
import os
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.metrics import accuracy_score
from joblib import dump, load
import warnings

# Ignore FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# import classes
from preprocessing.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from preprocessing.BagOfWords import BagOfWords
from preprocessing.TFIDF import TFIDF

from unsupervised.GaussianMixture import GaussianMixture
from unsupervised.KMeans import KMeans

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
  from preprocessing.PreProcess import PreProcess

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
# tfidf.find_optimal_params(X_train_clean_preprocess, y_train_clean)
X_train_clean_tfidf = tfidf.fit(X_train_clean_preprocess)
X_train_tfidf = tfidf.fit(X_train_preprocess)
X_val_tfidf = tfidf.fit(X_val_preprocess)
X_test_tfidf = tfidf.fit(X_test_preprocess)
print("** TF-IDF Completed **")

pca = PrincipalComponentAnalysis(50) # Principle Component Analysis
X_train_bag_pca = pca.fit(X_train_bag)
# pca.elbow_graph()
X_train_bag_pca = pca.fit(X_train_bag)
X_val_bag_pca = pca.fit(X_val_bag)
X_train_tfidf_pca = pca.fit(X_train_tfidf)
X_val_tfidf_pca = pca.fit(X_val_tfidf)
# X_train_clean_tfidf_pca = pca.fit(X_train_clean_tfidf)
print("** PCA Completed **")

# --- Unsupervised Learning ---
gmm = GaussianMixture(n_components=5, random_state=42) # Gaussian Mixture

### TFIDF -> GMM ###
gmm_tfidf_model_path = 'models/gmm_tfidf_model.joblib'
if os.path.exists(gmm_tfidf_model_path): # load model
  gmm = load(gmm_tfidf_model_path)
  print("** GMM model loaded from file **")
else: # train model
  gmm.fit(X_train_tfidf_pca)
  dump(gmm, gmm_tfidf_model_path)
  print("** GMM model trained and saved to file **")

labels = gmm.predict(X_train_tfidf_pca).values.flatten()

unlabeled_indices = np.where(y_train == -100)[0] # Indices of unlabeled data
y_train_tfidf_gmm = y_train.copy()
y_train_tfidf_gmm[unlabeled_indices] = labels[unlabeled_indices]

### Bag of Words -> GMM ###
gmm_bag_model_path = 'models/gmm_bag_model.joblib'
if os.path.exists(gmm_bag_model_path): # load model
  gmm = load(gmm_bag_model_path)
  print("** GMM model loaded from file **")
else: # train model
  gmm.fit(X_train_bag_pca)
  dump(gmm, gmm_bag_model_path)
  print("** GMM model trained and saved to file **")

labels = gmm.predict(X_train_bag_pca).values.flatten()

unlabeled_indices = np.where(y_train == -100)[0] # Indices of unlabeled data
y_train_bag_gmm = y_train.copy()
y_train_bag_gmm[unlabeled_indices] = labels[unlabeled_indices]

print("** GMM Completed **")

kmeans = KMeans(n_clusters=5, random_state=0) # K-Means

### TF-IDF -> KMeans ###
kmeans_tfidf_model_path = 'models/kmeans_tfidf_model.joblib'
if os.path.exists(kmeans_tfidf_model_path): # load model
  kmeans = load(kmeans_tfidf_model_path)
  print("** KMeans model loaded from file **")
else: # train model
  kmeans.fit(X_train_tfidf)
  dump(kmeans, kmeans_tfidf_model_path)
  print("** KMeans model trained and saved to file **")

labels = kmeans.predict(X_train_tfidf).values.flatten()

unlabeled_indices = np.where(y_train == -100)[0] # Indices of unlabeled data
y_train_tfidf_kmeans = y_train.copy()
y_train_tfidf_kmeans[unlabeled_indices] = labels[unlabeled_indices]

### Bag of Words -> KMeans ###
kmeans_bag_model_path = 'models/kmeans_bag_model.joblib'
if os.path.exists(kmeans_bag_model_path): # load model
  kmeans = load(kmeans_bag_model_path)
  print("** KMeans model loaded from file **")
else: # train model
  kmeans.fit(X_train_bag)
  dump(kmeans, kmeans_bag_model_path)
  print("** KMeans model trained and saved to file **")

labels = kmeans.predict(X_train_bag).values.flatten()

unlabeled_indices = np.where(y_train == -100)[0] # Indices of unlabeled data
y_train_bag_kmeans = y_train.copy()
y_train_bag_kmeans[unlabeled_indices] = labels[unlabeled_indices]

# --- Supervised Learning --- #

logr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # Logistic Regression

### TFIDF -> Logistic Regression ###
logr_tfidf_model_path = 'models/logr_tfidf_model'
if os.path.exists(logr_tfidf_model_path): # load model
  logr = load(logr_tfidf_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_clean_tfidf, y_train_clean)
  dump(logr, logr_tfidf_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = logr.predict(X_val_tfidf)
print("LOGISTIC REGRESSION (TF-IDF)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> Logistic Regression ###
logr_bag_model_path = 'models/logr_bag_model'
if os.path.exists(logr_bag_model_path): # load model
  logr = load(logr_bag_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_clean_bag, y_train_clean)
  dump(logr, logr_bag_model_path)
  print("** Logistic Regression model trained and saved to file **")
predictions = logr.predict(X_val_bag)
print("LOGISTIC REGRESSION (Bag of Words)")
print(accuracy_score(y_val, predictions))

### TFIDF -> GMM -> Logistic Regression ###
logr_tfidf_gmm_model_path = 'models/logr_tfidf_gmm_model'
if os.path.exists(logr_tfidf_gmm_model_path): # load model
  logr = load(logr_tfidf_gmm_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_tfidf_pca, y_train_tfidf_gmm)
  dump(logr, logr_tfidf_gmm_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = logr.predict(X_val_tfidf_pca)
print("LOGISTIC REGRESSION (TF-IDF + GMM)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> GMM -> Logistic Regression ###
logr_bag_gmm_model_path = 'models/logr_bag_gmm_model'
if os.path.exists(logr_bag_gmm_model_path): # load model
  logr = load(logr_bag_gmm_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_bag_pca, y_train_bag_gmm)
  dump(logr, logr_bag_gmm_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = logr.predict(X_val_bag_pca)
print("LOGISTIC REGRESSION (Bag of Words + GMM)")
print(accuracy_score(y_val, predictions))

### TFIDF -> KMEANS -> Logistic Regression ###
logr_tfidf_kmeans_model_path = 'models/logr_tfidf_kmeans_model'
if os.path.exists(logr_tfidf_kmeans_model_path): # load model
  logr = load(logr_tfidf_kmeans_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_tfidf, y_train_tfidf_kmeans)
  dump(logr, logr_tfidf_kmeans_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = logr.predict(X_val_tfidf)
print("LOGISTIC REGRESSION (TF-IDF + KMeans)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> KMEANS -> Logistic Regression ###
logr_bag_kmeans_model_path = 'models/logr_bag_kmeans_model'
if os.path.exists(logr_bag_kmeans_model_path): # load model
  logr = load(logr_bag_kmeans_model_path)
  print("** Logistic Regression model loaded from file **")
else:
  logr.fit(X_train_bag, y_train_bag_kmeans)
  dump(logr, logr_bag_kmeans_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = logr.predict(X_val_bag)
print("LOGISTIC REGRESSION (Bag of Words + KMeans)")
print(accuracy_score(y_val, predictions))

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
predictions = knn.predict(X_val_tfidf)
print("KNN (TF-IDF)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> KNN ###
knn_bag_model_path = 'models/knn_bag_model'
if os.path.exists(knn_bag_model_path): # load model
  knn = load(knn_bag_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_clean_bag, y_train_clean)
  dump(knn, knn_bag_model_path)
  print("** KNN model trained and saved to file **")
predictions = knn.predict(X_val_bag)
print("KNN (TF-IDF)")
print(accuracy_score(y_val, predictions))

### TFIDF -> GMM -> KNN ###
knn_tfidf_gmm_model_path = 'models/knn_tfidf_gmm_model'
if os.path.exists(knn_tfidf_gmm_model_path): # load model
  knn = load(knn_tfidf_gmm_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_tfidf_pca, y_train_tfidf_gmm)
  dump(knn, knn_tfidf_gmm_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = knn.predict(X_val_tfidf_pca)
print("KNN (TF-IDF + GMM)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> GMM -> KNN ###
knn_bag_gmm_model_path = 'models/knn_bag_gmm_model'
if os.path.exists(knn_bag_gmm_model_path): # load model
  knn = load(knn_bag_gmm_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_bag_pca, y_train_bag_gmm)
  dump(knn, knn_bag_gmm_model_path)
  print("** KNN model trained and saved to file **") 
predictions = knn.predict(X_val_bag_pca)
print("KNN (Bag of Words + GMM)")
print(accuracy_score(y_val, predictions))

### TFIDF -> KMeans -> KNN ###
knn_tfidf_kmeans_model_path = 'models/knn_tfidf_kmeans_model'
if os.path.exists(knn_tfidf_kmeans_model_path): # load model
  knn = load(knn_tfidf_kmeans_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_tfidf, y_train_tfidf_kmeans)
  dump(knn, knn_tfidf_kmeans_model_path)
  print("** Logistic Regression model trained and saved to file **") 
predictions = knn.predict(X_val_tfidf)
print("KNN (TF-IDF + KMeans)")
print(accuracy_score(y_val, predictions))

### Bag of Words -> KMeans -> KNN ###
knn_bag_kmeans_model_path = 'models/knn_bag_kmeans_model'
if os.path.exists(knn_bag_kmeans_model_path): # load model
  knn = load(knn_bag_kmeans_model_path)
  print("** KNN model loaded from file **")
else:
  knn.fit(X_train_bag, y_train_bag_kmeans)
  dump(knn, knn_bag_kmeans_model_path)
  print("** KNN model trained and saved to file **") 
predictions = knn.predict(X_val_bag)
print("KNN (Bag of Words + KMeans)")
print(accuracy_score(y_val, predictions))

print("** KNN Completed **")
