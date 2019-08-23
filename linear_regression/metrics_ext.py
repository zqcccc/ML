# -*- coding: utf-8 -*-

###Task 1 完成评估函数的定义
##输入变量为预测值及真实值，返回值相应评估指标，只允许使用numpy和python标准库


#x1:课程中使用数据的真实值
#x2:课程中使用模型预测的预测值

import numpy as np

# MAE
def metrics_MAE(x1,x2):
    
    #你的定义
    x1 = np.array(x1)
    x2 = np.array(x2)
    score = 0
    for i in range(x1.size):
        score = score + abs(x1[i]-x2[i])
    
    score = score/x1.size
    return score

# MSE
def metrics_MSE(x1,x2):
    
    #你的定义
    x1 = np.array(x1)
    x2 = np.array(x2)
    score = 0
    for i in range(x1.size):
        score = score + pow(x1[i]-x2[i])
    
    score = score/x1.size
    
    return score
	
# R2
def metrics_R2(x1,x2):
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    score = 0
    for i in range(x1.size):
        score = score + pow(x1[i]-x2[i])
    
    score = np.sqrt(score/x1.size)
    
    
    return score
