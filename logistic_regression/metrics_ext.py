# -*- coding: utf-8 -*-

###Task 1 完成sigmoid函数的定义
##输入变量一个/多个数值，返回值为经历sigmoid函数之后的结果

import numpy as np

def Sigmoid_fn(x1):
    
    #你的定义
    if type(x1) == type([]):
        x1 = np.array(x1)

    sig_result = 1 / (1+np.exp(-x1))
    
    return sig_result

Sigmoid_fn(0)
