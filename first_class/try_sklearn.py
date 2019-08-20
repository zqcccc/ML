# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

data = pd.read_csv('vehicle.csv')

# temp = data.iloc[:,:2]
feature = np.array(data.iloc[:,0:2])
labels = data['label'].tolist()

from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(feature, labels, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=9)
model.fit(feature_train, label_train)

prediction = model.predict(feature_test)

from sklearn.metrics import classification_report

print(classification_report(label_test,
                            prediction,
                            target_names = ['car', 'truck'],
                            labels = ['car', 'truck'],
                            digits = 4))


import sklearn
sklearn.metrics.accuracy_score(prediction, label_test)
sklearn.metrics.precision_score(label_test, prediction, average='weighted')



def metrics_precision(x1,x2):
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    labels = np.unique(x1)
    tp = 0
    fp = 0
    res = {}
    for label in labels:
        for i in range(x1.size):
            if(x1[i] == label and x2[i] == label):
                tp = tp + 1
            elif(x1[i] != label and x2[i] == label):
                fp = fp + 1
        res[label] = tp / (tp + fp)
        tp = 0
        fp = 0
        
    return res
    
precision = metrics_precision(label_test, prediction)

def metrics_recall(x1,x2):
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    labels = np.unique(x1)
    tp = 0
    fn = 0
    res = {}
    for label in labels:
        for i in range(x1.size):
            if(x1[i] == label and x2[i] == label):
                tp = tp + 1
            elif(x1[i] == label and x2[i] != label):
                fn = fn + 1
        res[label] = tp / (tp + fn)
        tp = 0
        fn = 0
        
    return res

recall = metrics_recall(label_test, prediction)



def distance_f1_score(x1,x2):
    
    x1 = np.array(x1)
    labels = np.unique(x1)
    res = {}
    
    precision = metrics_precision(x1, x2)
    recall = metrics_recall(x1, x2)
    
    for label in labels:
        res[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
    
    
    return res

f1_score = distance_f1_score(label_test, prediction)
