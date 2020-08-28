# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:07:18 2018

@author: Amey Ambaji
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importing the dataset
dataset = pd.read_csv("2017_DATA_SA_Crash.csv")
x = dataset.iloc[:,[13,14,19,20,21,22,29,24,5]].values
y = dataset.iloc[:,26].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_3 = LabelEncoder()
x[:, 2] = labelencoder_x_3.fit_transform(x[:, 2])

labelencoder_x_4 = LabelEncoder()
x[:, 3] = labelencoder_x_4.fit_transform(x[:, 3])

labelencoder_x_5 = LabelEncoder()
x[:, 4] = labelencoder_x_5.fit_transform(x[:, 4])

labelencoder_x_6 = LabelEncoder()
x[:, 5] = labelencoder_x_6.fit_transform(x[:,5])

labelencoder_x_7 = LabelEncoder()
x[:, 6] = labelencoder_x_7.fit_transform(x[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [2])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 2:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test,y_pred))