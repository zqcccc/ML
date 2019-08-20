# -*- coding: utf-8 -*-

###Task 1 完成评估函数的定义
##输入变量为预测值及真实值，返回值相应评估指标，不允许使用除数据结构外的第三方库
#precision

import numpy as np
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

#recall


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
	
#f1_score



def distance_f1_score(x1,x2):
    
    x1 = np.array(x1)
    labels = np.unique(x1)
    res = {}
    
    precision = metrics_precision(x1, x2)
    recall = metrics_recall(x1, x2)
    
    for label in labels:
        res[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
    
    
    return res
