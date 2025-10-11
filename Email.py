# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 21:40:57 2025

@author: user
"""

# ==============================
# 1. Imports
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ==============================
# 2. Load and Clean Data
# ==============================
df = pd.read_csv("spam.csv",encoding= 'latin1')
#create copy of dataset
df1 = df.copy()
#check whether the dataset is properly loaded
print(df1.head())
'''   v1                                                 v2 Unnamed: 2
0   ham  Go until jurong point, crazy.. Available only ...        NaN   
1   ham                      Ok lar... Joking wif u oni...        NaN   
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   
3   ham  U dun say so early hor... U c already then say...        NaN   
4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   

  Unnamed: 3 Unnamed: 4  
0        NaN        NaN  
1        NaN        NaN  
2        NaN        NaN  
3        NaN        NaN  
4        NaN        NaN  '''
#=================================
"""
EXPLORATORY DATA ANALYSIS
1.Getting to know the data
2.Data processing(missing values)
3.Cross tables and data visualization
"""
print(df1.info())
print("Total Coloumns with null values:\n", df1.isnull().sum())
# use cross tables(optional in this case because only two relevant columns)
v1_v2 = pd.crosstab(index= df1["v1"], columns= df1['v2'], margins=True,normalize= 'index')
print(v1_v2)
#taking only the relevant coloumns into consideration
df1 = df1[['v1','v2']]
df1.columns = ['label','message']
#Convert class label to binary
df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

X = df1['message']
y = df1['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorized shape:", X_train_vec.shape)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# Evaluate nb
y_pred = nb.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Naive Bayes Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Prediction function
def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = nb.predict(vec)[0]
    prob = nb.predict_proba(vec)[0].max()
    return "SPAM" if pred == 1 else "HAM", f"{prob:.2%}"

# Test
test_msg = input("Enter a message you would like to classify:").lower()
label, confidence = predict_spam(test_msg)
print(f"\nMessage: {test_msg}")
print(f"Prediction: {label} (Confidence: {confidence})")
