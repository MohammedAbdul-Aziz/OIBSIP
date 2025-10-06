# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:22:25 2025

@author: user
"""
#-----------------------------
# Problem Statement:IRIS FLOWER CLASSIFICATION
#Iris flower has three species; setosa, versicolor, and virginica, which differs according to their
#measurements. Now assume that you have the measurements of the iris flowers according to
#their species, and here your task is to train a machine learning model that can learn from the
#measurements of the iris species and classify them.
#-----------------------------
# Importing Files 
#-----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
#-----------------------------------------------------------
#Importing data
data_iris = pd.read_csv("D:\Oasis Intern\iris.csv")
#creating a copy of data
data = data_iris.copy()
#Check if importing is complete
print(data.head())

"""
EXPLORATORY DATA ANALYSIS
1.Getting to know the data
2.Data processing(missing values)
3.Cross tables and data visualization

"""

print(data.info())
print("Data columns with null values:\n",data.isnull().sum())
#No null values Detected
#Summary of Numerical data
summary_num = data.describe()
print(summary_num)
"""
               Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
count  150.000000     150.000000    150.000000     150.000000    150.000000
mean    75.500000       5.843333      3.054000       3.758667      1.198667
std     43.445368       0.828066      0.433594       1.764420      0.763161
min      1.000000       4.300000      2.000000       1.000000      0.100000
25%     38.250000       5.100000      2.800000       1.600000      0.300000
50%     75.500000       5.800000      3.000000       4.350000      1.300000
75%    112.750000       6.400000      3.300000       5.100000      1.800000
max    150.000000       7.900000      4.400000       6.900000      2.500000
From the above results we can observe that the mean and median does not differ much ,
With this theory we can confirm that either mean or median can be used to fill abnormal values
Less standar deviation indicates that our dataset is near to perfect"""
#Summary of categorical values
summary_cat = data.describe(include='O')
print(summary_cat)
"""
            Species
count           150
unique            3
top     Iris-setosa
freq             50
This shows that the dataset is perfectly balanced which is an ideal situation for a dataset used in machine learning.
It logically follows that each of the three species is represented by exactly 50 flowers (50×3=150)"""
#frequency of each categories,Find special charecters
print(data['SepalLengthCm'].value_counts())
print(data['SepalWidthCm'].value_counts())
print(data['PetalLengthCm'].value_counts())
print(data['PetalWidthCm'].value_counts())
print(data['Species'].value_counts())
#No special Chareters were found in anyof the columns

#DATA PRE-PROCESSING

print(data.isnull().sum())
missing = data[data.isnull().any(axis=1)]#Checking coloumn missing values
print(missing)
#No missing valuies were found in any columns
#Relationship between independent variables
# Select only numeric columns
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print(correlation)
"""
                     Id  SepalLengthCm  ...  PetalLengthCm  PetalWidthCm
Id             1.000000       0.716676  ...       0.882747      0.899759
SepalLengthCm  0.716676       1.000000  ...       0.871754      0.817954
SepalWidthCm  -0.397729      -0.109369  ...      -0.420516     -0.356544
PetalLengthCm  0.882747       0.871754  ...       1.000000      0.962757
PetalWidthCm   0.899759       0.817954  ...       0.962757      1.000000

[5 rows x 5 columns]"""
#All the values are near to 1,Which indicates that the data is strongly correlated
#consider categorical variables
print(data.columns)
#preparing cross tables and data visualization
#Species proportion table
species = pd.crosstab(index = data["Species"], columns = 'count',normalize=True)
print(species)
#Species vs SepalLengthCm
Species_Sepallen= pd.crosstab(index = data["Species"], columns = data['SepalLengthCm'],margins=True,normalize='index')
print(Species_Sepallen)
#Iris-virginica and Iris-versicolor has few to no flowers of smaller Sepal length but have a good proportion of flowers of larger Sepal length
#Where as Iris-setosa is quiet the opposite of both
#Species vs SepalWidthCm
Species_Sepalwid= pd.crosstab(index = data["Species"], columns = data['SepalWidthCm'],margins=True,normalize='index')
print(Species_Sepalwid)
#Iris-virginica and Iris-versicolor has a good proportion of flowers of smaller Sepal Width but have few or no flowers of larger Sepal Width
#Where as Iris-setosa is quiet the opposite of both
#Species vs PetalLengthCm 
Species_Petallen= pd.crosstab(index = data["Species"], columns = data['PetalLengthCm'],margins=True,normalize='index')
print(Species_Petallen)
#Iris-setosa has a smaller Petal lenth,Iris-versicolor  has an average petal length,Iris-virginica has a larger petal length
#Species vs PetalWidthCm 
Species_Petalwid= pd.crosstab(index = data["Species"], columns = data['PetalWidthCm'],margins=True,normalize='index')
print(Species_Petalwid)
#Iris-setosa has a smaller Petal Width,Iris-versicolor  has an average petal Width,Iris-virginica has a larger petal Width
#The above are column correlations with respect to the output variable
#LOGISTIC REGRESSION 
data['Species']=data['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
print(data['Species'].unique())
data = data.drop('Id',axis = 1)
#Dropping the axis column because it has no significance on the output
columns_list=list(data.columns)
print(columns_list)
#Seperating the input names from data;Getting independent variables
features = list(set(columns_list)-set(['Species']))
print(features)
#storing the output values into y
y=data['Species'].values
print(y)
#storing the values from input features
x=data[features].values
print(x)
#Splitting the data into train and test'
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)
#Scale the features using StandardScaler
scaler = StandardScaler()
# Fit scaler ONLY on training data
scaler.fit(train_x)
# Transform both train and test data
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
#making an instance of the model
logistic = LogisticRegression()
#Fitting the values for x and y
logistic.fit(train_x_scaled, train_y)
print(logistic.coef_)
print(logistic.intercept_)
#prediction from test data
prediction = logistic.predict(test_x_scaled)
print(prediction)
#Evaluating the model
#confusion matrix
CM=confusion_matrix(test_y,prediction)
print(CM)
#accuracy score
accuracy = accuracy_score(test_y,prediction)
print(accuracy)
#We Got a High Accuracy of 97.8%
#printing misclassified values from predictions
print("Misclassified samples : %d" %(test_y != prediction).sum())
#Only 1 sample was Misclassified

#----------------------------------------------------------------
#KNN
#----------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(train_x,train_y)
prediction2 = KNN.predict(test_x)
print(prediction2)
#Confusion Matrix
CM=confusion_matrix(test_y,prediction2)
print(CM)
#accuracy score
accuracy2 = accuracy_score(test_y,prediction2)
print(accuracy2)
#We got the same accuracy as Logistic regression , i.e 97.8%
print("Misclassified samples : %d" %(test_y != prediction2).sum())
#Only 1 sample was Misclassified
#calculating error for k value between 1 and 20
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    print("Misclassified samples : %d" %(test_y != pred_i).sum())
#We Get Only 1 misclassified value for all the values of k between 1-20
#This is a special case where even K=1 gives the same accuracy as k=20
