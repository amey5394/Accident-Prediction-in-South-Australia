## **Supervised Machine Learning Project: Accident Prediction in South Australia**

This project was focused on the road accidents in South Australia. A model to predict the severity of an accident for a location based on several parameters like road type, Alcohol contains, Drugs, Gender, Weather, Age, etc. The primary goal was to reduced the road accidents by providing data to the local authorities and decision makers. This information is useful for tourist and local pedestrians to comprehend the location in a better way and ensure safety measure while travelling.

The database used is from [data.sa](data.sa) (open source govt. website) data uploaded is to track region wise past accidents.

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for


### Data Exploration
Let's start with importing the necessary libaries, reading in the data, and checking out the dataset.
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('2017_DATA_SA_Crash_ANN.csv')
X = dataset.iloc[:, [1,5,14,19,20,21,22,29,24,11,8]].values
y = dataset.iloc[:, 26].values
```
The data set sonsists of some parameters based on which I have prepared a model to predict the road accidents



### Preparing Data
Prior using the data for machine learning algorithms, it often must be cleaned, normalised or formatted — this is known as data preprocessing. As this dataset was uploaded on the government website to keep a track of the previous accidents the data was pretty much clean and normalised only certain location names were to be adjusted. This preprocessing can help with the outcome and predictive capability of almost all types of learning algorithms.

```python
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])

labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])

labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])

labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])

labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
```
Sklearn provides a very efficient tool for encoding the levels of categorical features into numeric values. LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels.

```python

```
