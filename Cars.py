# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 18:44:26 2025

@author: user
"""
#-----------------------------
# Problem Statement:Car Price Prediction with Machine Learning
#The price of a car depends on a lot of factors like the goodwill of the brand of the car,
#features of the car, horsepower and the mileage it gives and many more. Car price
#prediction is one of the major research areas in machine learning. So if you want to learn
#how to train a car price prediction model then this project is for you.

#-----------------------------
# Importing Files 
#-----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------------------------------------
#Importing data
data_cars = pd.read_csv('D:\Oasis Intern\CARS\car data.csv')
#creating a copy of data
data = data_cars.copy()
#Checking if importing is complete
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
              Year  Selling_Price  Present_Price     Driven_kms       Owner
count   301.000000     301.000000     301.000000     301.000000  301.000000
mean   2013.627907       4.661296       7.628472   36947.205980    0.043189
std       2.891554       5.082812       8.642584   38886.883882    0.247915
min    2003.000000       0.100000       0.320000     500.000000    0.000000
25%    2012.000000       0.900000       1.200000   15000.000000    0.000000
50%    2014.000000       3.600000       6.400000   32000.000000    0.000000
75%    2016.000000       6.000000       9.900000   48767.000000    0.000000
max    2018.000000      35.000000      92.600000  500000.000000    3.000000"""

#From the above summary we can see that the data is symmetrically distributed

#Summary of categorical values
summary_cat = data.describe(include='O')
print(summary_cat)
print(data.columns)
#frequency of each categories,Find special charecters
#pd.set_option('display.max_rows', None)
print(data['Car_Name'].value_counts())
print(data['Year'].value_counts())
print(data['Selling_Price'].value_counts())
print(data['Present_Price'].value_counts())
print(data['Driven_kms'].value_counts())
print(data['Fuel_Type'].value_counts())
print(data['Selling_type'].value_counts())
print(data['Transmission'].value_counts())
print(data['Owner'].value_counts())
#No unique value detected in any of the columns

#checking for duplicate values
data[data.duplicated()]
#Duplicates found
data.drop_duplicates()
#Dropping duplicates from the dataset
#Normalized value counts (percentage)
print(data.columns.value_counts(normalize=True))

#DATA VISUALIZATION:
    
#Histogram for numerical columns
data.hist(figsize=(10, 5))
"""Year:The data is left-skewed. This shows that most of the cars in the dataset are relatively new,with a large concentration of cars manufactured in 2015 or later. There are very few older cars.
Selling_Price:This distribution is right-skewed. Most cars have a selling price under 5 lakhs,with very few cars having a high resale value.
Present_Price:Similar to Selling_Price, this is also right-skewed.The majority of cars have an original showroom price under 20 lakhs, while very expensive cars are rare in this dataset.
Driven_Kms:This is heavily right-skewed. The vast majority of cars have been driven less than 100,000 kilometers,with a few cars having been driven a very high number of kilometers.
Owner:This is not a continuous distribution but a count. The histogram shows that the overwhelming majority of cars have had 0 previous owners (i.e., they are first-hand). Very few have had 1, and almost none have had 2 or 3."""

#Box plots for outlier detection
data.boxplot(figsize=(10,5))
"""Year, Selling_Price, Present_Price, Owner: These features have very compressed box plots, indicating that the majority of their data is clustered within a narrow range. They each show a few high-value outliers.
Driven_kms:The box itself is relatively small, showing that 50% of the cars have a similar range of kilometers. However, there are numerous outliers with very high values, indicating a few cars that have been driven far more than the typical car in this dataset."""
#Box plot for selling price
data["Selling_Price"].plot(kind="box")
#The plot is right skewed.Outliers were detected.
#Box plot for present price
data["Present_Price"].plot(kind="box")
#The plot is also right skewed.A few outliers especially one near 90 lakhs were detected.

#Scatter plot between Selling Price and Present Price to find te relation between them
data.plot(kind="scatter", x="Present_Price", y="Selling_Price")
#A strong positive correlation is observed between the two variables
#This may result in a strong factor while preparing the model.

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
                   Year  Selling_Price  Present_Price  Driven_kms     Owner
Year           1.000000       0.236141      -0.047192   -0.524342 -0.182104
Selling_Price  0.236141       1.000000       0.878914    0.029187 -0.088344
Present_Price -0.047192       0.878914       1.000000    0.203618  0.008058
Driven_kms    -0.524342       0.029187       0.203618    1.000000  0.089216
Owner         -0.182104      -0.088344       0.008058    0.089216  1.000000"""
#The above correlation suggests:
    # Present_Price and Selling_Price has a very high correlation
    #Driven_kms,Owner have a very low negetive correlation
    #Driven_kms and Owner has a moderate negetive correlation
#consider categorical variables
print(data.columns)
#preparing cross tables and data visualization
#Selling_Price proportion table
Selling = pd.crosstab( index = data["Selling_Price"], columns = 'count',normalize=True) 
print(Selling) 
#At this point Nothing can be interpreted using the Selling table only, Use other factors with Selling_Price to find important affecting factors
#Selling_Price vs Car_Name
#Creates a new column in the dataset named "Brand",Which contains only the brand from the car name
data["Brand"] = data["Car_Name"].apply(lambda x : x.split()[0])
data = data.drop("Car_Name",axis = 1)
#Using a boxplot
plt.figure(figsize=(15,8))
sns.boxplot(x='Selling_Price', y='Brand', data=data)
plt.xlabel("Selling Price")
plt.ylabel("Car Brand")
plt.title("Distribution of Selling Prices by Brand")
plt.show()
"""High-Value Brands(Fortuner,innova etc) have boxes situated far to the right , ehich mans they have high median selling price
Wide-Price range Brands like fortuner has a very wide box which means its elling prices are very spread out.Brands like Yamaha has a very narrow box because of their small rnge of prices
Low-Value Brands(TVS,Bajaj etc) have boxes clustered on the far left,Showing they have a low selling price
"""
#Using a bar plot
brand_price = data.groupby('Brand')["Selling_Price"].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=brand_price.values, y=brand_price.index)
plt.xlabel("Average Selling Price")
plt.ylabel("Car Brand")
plt.title("Average Selling Price by Car Brand")
plt.show()
"""As seen in the graph,Land cruiser has a very high selling value,followed by fortuner,innova etc..
Brands like Activa,Hero and TVS has low avg selling price
"""
#Selling_price vs Year
data["Car_Age"]=2025 - data['Year']
data.drop("Year",axis=1,inplace=True)
#Using a ScatterPlot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Car_Age', y='Selling_Price', data=data)
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.title("Selling Price vs. Car Age")
plt.grid(True)
plt.show()
#With this plot,a conclusion can be made that there is a negetive correlation between these two variables.

#Selling_Price vs Present_Price

#Using a ScatterPlot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data)
plt.xlabel("Present_Price")
plt.ylabel("Selling Price")
plt.title("Selling Price vs. Present_Price")
plt.grid(True)
plt.show()
#We already saw their relation

#Selling_Price vs Driven_kms

#Using a ScatterPlot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=data)
plt.xlabel("Driven_kms")
plt.ylabel("Selling Price")
plt.title("Selling Price vs. Driven_kms")
plt.grid(True)
plt.show()
#A right skewness is observed with a few outliers

#Selling_Price vs Fuel_Type

#Using a boxplot
plt.figure(figsize=(15,8))
sns.boxplot(x='Selling_Price', y='Fuel_Type', data=data)
plt.xlabel("Selling Price")
plt.ylabel("Fuel_Type")
plt.title("Distribution of Selling Prices by Fuel_Type")
plt.show()
#It is observed that:
    #Petrol is left clustered which means the range of selling price is low to average,a few outliers are also observed
    #Diesel has a wide range of selling prices with a high median selling price than petrol and much high max price,a few outliers are also observed
    #CNG is an interesting case because only a few cars of CNG are available which has a very small range of selling price

#Selling_Price vs Selling_type

#Using a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x="Selling_type", y="Selling_Price",data = data)
plt.xlabel("Selling_type")
plt.ylabel("Selling_Price")
plt.title("Average Selling Price by Selling_type")
plt.show()
#The plot shows that the cars sold by Dealers are significantly expensive than the cars sold by Individuals

#Using a boxplot
plt.figure(figsize=(15,8))
sns.boxplot(x='Selling_Price', y='Selling_type', data=data)
plt.xlabel("Selling Price")
plt.ylabel("Selling_type")
plt.title("Distribution of Selling Prices by Selling_type")
plt.show()
#This boxplot confirms the above statement by depicting that the median of Dealers is higher that Max of Individuals

#Selling_Price vs Transmission

#Using a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x="Transmission", y="Selling_Price",data = data)
plt.xlabel("Transmission")
plt.ylabel("Selling_Price")
plt.title("Average Selling Price by Transmission")
plt.show()


#Using a boxplot
plt.figure(figsize=(15,8))
sns.boxplot(x='Selling_Price', y='Transmission', data=data)
plt.xlabel("Selling Price")
plt.ylabel("Transmission")
plt.title("Distribution of Selling Prices by Transmission")
plt.show()

#From the above two plots we can see that Automatic cars cover a wider range of selling price than manual cars

#Selling_Price vs Owner

#Using a ScatterPlot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Owner', y='Selling_Price', data=data)
plt.xlabel("Owner")
plt.ylabel("Selling Price")
plt.title("Selling Price vs. Owner")
plt.grid(True)
plt.show()
#From the plot we can see that there is a negetive correlation between the two variables. As the number of owners increases,Selling price decreases

#DATA PREPROCESSING

scaler = StandardScaler()
#This scales the values to have mean = 0 and standard deviation = 1 to better distribution
data[['Selling_Price','Present_Price','Driven_kms']] = scaler.fit_transform(data[['Selling_Price','Present_Price','Driven_kms']])
#Handling categorical values
data= pd.get_dummies(data, columns=['Fuel_Type', 'Selling_type', 'Transmission',], drop_first=True)
data= pd.get_dummies(data, columns=['Brand'], drop_first=True)
#Outlier Handling
#Driven_kms outliers are being fixed because it contains the maximum outliers and the rest ofthe varibales are replaced with dummy variables
Q1 = data['Driven_kms'].quantile(0.25)
Q3 = data['Driven_kms'].quantile(0.75)
IQR = Q3 - Q1
data= data[~((data['Driven_kms'] < (Q1 - 1.5 * IQR)) | (data['Driven_kms'] > (Q3 + 1.5 * IQR)))]
#A total of 301-293 = 8 rows were removed , the method followed to remove outliers was interquantile range
print(data.columns)

#Creating a ML Model

# X contains all the predictive features
X = data.drop('Selling_Price', axis=1) 
# y is the target variable we want to predict
y = data['Selling_Price']

#Splitting the data so we can train the model and test it on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Make Predictions on the Test Data
predictions = model.predict(X_test)

#R-squared score
r2 = metrics.r2_score(y_test, predictions)
print("R-squared Score: {r2:.2%}")
#R-squared Score: 90.39%

#Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE): {mae:,.2f}")
#Mean Absolute Error (MAE): 0.24