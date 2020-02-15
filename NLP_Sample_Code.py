#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Natural Language Processing


# In[48]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[49]:


# Importing the dataset
dataset = pd.read_csv('D:DS_TriS/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[50]:


dataset.head()


# In[51]:



# Cleaning the texts


# In[52]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[53]:


corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[54]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[55]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[56]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[57]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred


# In[58]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[59]:


TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP,TN,FP,FN


# In[60]:


Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy
#73% is very good accuracy as we have very less almost 1000 rows of data.
#In a very large dataset it will improve or will give better accuracy result.


# In[61]:


Precision = TP / (TP + FP)
Precision


# In[62]:


Recall = TP / (TP + FN)
Recall


# In[63]:


F1_Score = 2 * Precision * Recall / (Precision + Recall)
F1_Score


# In[64]:


#We can also use other machine learning model here in place of Naive Bayes, to get better result we have to focus on all the values of
#Accuracy, Precision, Recall, F1_score.


# In[ ]:




