#!/ Users/ python3 
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Data Import
df = pd.read_csv("data.csv")

data = df.dropna()

data_np = data.values
X = data_np[:, 1:].astype(np.float64)
y = data_np[:, 0].astype(np.int64)

#  SparsePCA
from sklearn.decomposition import SparsePCA
pca = SparsePCA(n_components=5)
pca.fit(X)

# Transform PCA
X_r = pca.fit(X).transform(X)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X_r, y, train_size=0.3)

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# model fit
model.fit(X_train, y_train)

# accuracy
model.score(X_test, y_test)





