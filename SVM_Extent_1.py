# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:38:58 2018

@author: Amey Ambaji
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('2017_DATA_SA_Casualty.csv')
x1 = dataset.iloc[:, [5,8]].values
X = dataset.iloc[:, [5,4,3,7,10]].values
y = dataset.iloc[:, 8].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, :1])
X[:, :1] = imputer.transform(X[:,:1])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import seaborn as sns
plt.figure(figsize=(15,10))
sns.barplot(x=x1[], y=x1['1'])
plt.xticks(rotation= 45)
plt.xlabel('Age')
plt.ylabel('Purchase')
plt.title('Purchase Given Ages')
plt.show()


sns.boxplot(x="Casualty Type", y="AGE", hue="Sex", data=dataset, palette="PRGn")
plt.show()