import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from __future__ import unicode_literals
import ast # this is just use to evaluate the lemmas
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

decisionTree = DecisionTreeClassifier()
ts_test = []

def Train():
    ## Classifications are under Sentiment: 0=neg and 1=pos
    ## SetimentText==Tweets

    training_set = pd.read_csv("train.csv", index_col=False, encoding='latin-1', header=0)
    ts_new = training_set[:80000]
    ts_test = training_set[~training_set.ItemID.isin(ts_new.ItemID)]
                           
    vectorizervectori  = TfidfVectorizer() # vectorizes and normalizes the data

    features = vectorizer.fit_transform(ts_new.lemma.astype('unicode')) #assumes a sparse matrix, prepares features
    #print(vectorizer.vocabulary_)
    smatrix = vectorizer.transform(ts_test.lemma.astype('unicode')) #puts test data in a form acceptable by the tree
    #print(smatrix)

    # these both utilize the training set
    y = ts_new['Sentiment']
    X = features
    
    decisionTree = DecisionTreeClassifier(min_samples_split=20, random_state=99)

    # Trains the tree
    decisionTree = decisionTree.fit(X,y)