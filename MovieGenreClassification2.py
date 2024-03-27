#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,classification_report
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import string
import nltk


# In[2]:


train_data = pd.read_csv('train_data.txt', header=None, sep=':::', names=['ID', 'Title', 'Genres', 'Description'], engine='python')
train_data.head()


# In[3]:


train_data.info()


# In[5]:


test_data = pd.read_csv('test_data.txt', sep=':::', names=['ID', 'Title', 'Description'], engine='python')
test_data.head()


# In[6]:


test_data.info()


# In[7]:


stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

def clean_description(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()
    return text


# In[8]:


train_data['Clean_Description'] = train_data['Description'].apply(clean_description)


# In[9]:


train_data.head()


# In[27]:


X = train_data['Clean_Description'].iloc[:20000]
y = train_data['Genres'].iloc[:20000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)


# In[28]:


tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.fit_transform(X_test)


# In[29]:


svm = SVC()
svm.fit(X_train_tfidf, y_train)


# In[30]:


svm.score(X_train_tfidf, y_train)


# In[ ]:




