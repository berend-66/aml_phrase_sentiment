# import packages

import os
import sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression as sk_OLS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import math

from sklearn.metrics import r2_score

from collections import Counter

# packages
from sklearn.metrics import mean_squared_error, r2_score, f1_score

# import classes
from preprocessing.PrincipalComponentAnalysis import PrincipalComponentAnalysis
from preprocessing.PreProcess import PreProcess
from preprocessing.BagOfWords import BagOfWords
from preprocessing.TFIDF import TFIDF
from supervised.LinearRegression import LinearRegression

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

# print(f"Train Data Shape: {X_train.shape}")
# print(f"Cleaned Train Data Shape: {train_data_clean['Phrase'].shape}")
# print(f"Validation Data Shape: {X_val.shape}")
# print(f"Test Data Shape: {X_test.shape}")

# --- Preprocessing Data ---
pre_processor = PreProcess()

X_train_preprocess = pre_processor.process(X_train)
X_train_clean_preprocess = pre_processor.process(X_train_clean)
X_val_preprocess = pre_processor.process(X_val)
X_test_preprocess = pre_processor.process(X_test)

combined_data = pd.concat([X_train_preprocess, X_val_preprocess, X_test_preprocess])
bag = BagOfWords(combined_data) # Bag of Words

n = 3000
X_train_bag = bag.bag_of_words(X_train_preprocess, threshold_m=n)
X_train_clean_bag = bag.bag_of_words(X_train_clean_preprocess, threshold_m=n)
X_val_bag = bag.bag_of_words(X_val_preprocess, threshold_m=n)
X_test_bag = bag.bag_of_words(X_test_preprocess, threshold_m=n)

tfidf = TFIDF() # TF-IDF
X_train_clean_tfidf = tfidf.fit(X_train_clean_preprocess)

pca = PrincipalComponentAnalysis(125) # Principle Component Analysis
X_train_clean_bag_pca = pca.fit(X_train_clean_bag)
X_train_clean_tfidf_pca = pca.fit(X_train_clean_tfidf)
# pca.elbow_graph()

# --- Supervised Learning ---
lr = LinearRegression() # Linear Regression
lr.fit(X_train_clean_bag_pca, y_train_clean)
prediction = lr.predict(X_train_clean_bag_pca)
print("LINEAR REGRESSION")
print(f1_score(y_train_clean, prediction, average='weighted'))