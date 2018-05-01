import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from __future__ import unicode_literals
import ast # this is just use to evaluate the lemmas
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


decisionTree = DecisionTreeClassifier()
vectorizer  = TfidfVectorizer()
randomForest = RandomForestClassifier()
ts_test = []
trainingSet = pd.read_csv("train_parsed.csv", index_col=0, encoding='latin-1', header=0)
tweets = pd.read_csv("tweets_parsed.csv", low_memory=False, index_col=0, dtype='object')

def DataSets(trainingSet,tweets):
    
    tsTrain = trainingSet[:90000]
    tsTest = trainingSet[~trainingSet.ItemID.isin(tsTrain.ItemID)]
    xTrain = vectorizer.fit_transform(tsTrain.lemma.astype('unicode')) 
    xTest = vectorizer.transform(tsTest.lemma.astype('unicode')) 
    yTrain,yTest = tsTrain['Sentiment'], tsTest['Sentiment']
    xTweet = vectorizer.transform(tweets.lemma.astype('unicode'))
    return xTrain, yTrain, xTest, yTest, xTweet


def TrainDecisionTree(xTrain,yTrain,xTest,yTest):
    ## Classifications are under Sentiment: 0=neg and 1=pos
    ## SetimentText==Tweets    
    t0 = time.time()
    decisionTree = decisionTree.fit(xTrain,yTrain)
    
    t1 = time.time()
    testPrediction = decisionTree.predict(xTest)
    t2 = time.time()
    print('Time taken to train '+str(t1-t0))
    print('Time taken to predict '+str(t2-t1))

    print(classification_report(yTest, testPrediction))
    
    return decisionTree
        
    
def Tree(xTweet, decisionTree):
    
    predictTweets = decisionTree.predict(xTweet)
    tweetProb = decisionTree.predict_proba(xTweet) # <- not using this, here in case we want it
    return predictTweets


def TrainForest(xTrain,yTrain,xTest,yTest):
    randomForest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    randomForest.fit(xTrain,yTrain)
    randomForestPrediction = randomForest.predict(xTest)
    accuracy = accuracy_score(yTest,randomForestPrediction)
    
    print(classification_report(xTest, randomForestPrediction))
    print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    
    return randomForest

def Forest(xTweet, randomForest):
    
        forestPredictTweets = randomForest.predict(xTweet)
        # this function returns an array that has the tweet sentiement in order
        # Can be appended to the original tweet dataframe for further analysis
        return forestPredictTweets
    
    
## Get to the below if we have time, it explores feature space
"""   
def TreeFeatures(decisionTree, y):
    
    ymin = min(vals-.05)
    ymax = max(vals+.05)
    vals = decisionTree.feature_importances_
    plt.plot(vals, 'o')
    plt.ylim(ymin,ymax)
    plt.show()

    return vals


def TreeFeaturesClose(vals, cutoff):
    
    ymin = min(vals-.05)
    ymax = max(vals+.05)
    vals = decisionTree.feature_importances_
    maps = {value: key for key, value in vectorizer.vocabulary_.items()}
    plt.plot(vals, 'o')
    plt.ylim(ymin,ymax)
    plt.show()
    
    def get(key, d=maps, default=None):

        if key in d:
            return d[key]
        else:
            return default


        abc = out_cut.x.apply(get)
"""
