# -*- coding: utf-8 -*-


###Task 2 使用sklearn解决分类问题
##输入数据为车辆数据vehicle
#要求：
#（1）分别使用k=3，5，9对目标数据集进行分类
#（2）可视化出分类结果，并标注被错误分类的点
#（3）呈现完整的可视化结果
#（4）计算f1值

########  你的可视化结果看起来应该类似于'answer_demo.png'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv('vehicle.csv')

feature = np.array(data.iloc[:,0:2])
labels = data['label'].tolist()

from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(feature, labels, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier




# def metrics_precision(x1,x2):
#
#     x1 = np.array(x1)
#     x2 = np.array(x2)
#     labels = np.unique(x1)
#     tp = 0
#     fp = 0
#     res = {}
#     for label in labels:
#         for i in range(x1.size):
#             if(x1[i] == label and x2[i] == label):
#                 tp = tp + 1
#             elif(x1[i] != label and x2[i] == label):
#                 fp = fp + 1
#         res[label] = tp / (tp + fp)
#         tp = 0
#         fp = 0
#
#     return res
#
# def metrics_recall(x1,x2):
#
#     x1 = np.array(x1)
#     x2 = np.array(x2)
#     labels = np.unique(x1)
#     tp = 0
#     fn = 0
#     res = {}
#     for label in labels:
#         for i in range(x1.size):
#             if(x1[i] == label and x2[i] == label):
#                 tp = tp + 1
#             elif(x1[i] == label and x2[i] != label):
#                 fn = fn + 1
#         res[label] = tp / (tp + fn)
#         tp = 0
#         fn = 0
#
#     return res
#
# def distance_f1_score(x1,x2):
#
#     x1 = np.array(x1)
#     labels = np.unique(x1)
#     res = {}
#
#     precision = metrics_precision(x1, x2)
#     recall = metrics_recall(x1, x2)
#
#     for label in labels:
#         res[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
#
#
#     return res

for k in [3, 5, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(feature_train, label_train)
    
    prediction = model.predict(feature_test)
    
    plt.plot()
    plt.scatter(data['length'][data['label'] == 'car'], data['width'][data['label'] == 'car'], c='#66ffff')
    plt.scatter(data['length'][data['label'] == 'truck'], data['width'][data['label'] == 'truck'], c='r')
    
    for x in range(len(prediction)):
        if prediction[x] != label_test[x]:
            plt.scatter(feature_test[x][0],feature_test[x][1], c='#000000')
    plt.show()

import metrics_ext
f1_score = metrics_ext.distance_f1_score(label_test, prediction)
print('f1_score:', f1_score)
