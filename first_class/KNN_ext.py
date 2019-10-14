# -*- coding: utf-8 -*-

###Task 1 完成距离度量函数的定义
##输入变量为两个向量，返回值为距离，不允许使用除数据结构外的第三方库
#曼哈顿距离
import numpy as np

def distance_manhattan(x1,x2):
    
    vec1 = np.array(x1)
    vec2 = np.array(x2)
    distance = np.sqrt(np.sum(vec1 - vec2) ** 2)
    return distance


#欧式距离
def distance_euclidean(x1,x2):
    
    vec1 = np.array(x1)
    vec2 = np.array(x2)
    distance= np.sqrt(np.sum(np.square(vec1 - vec2)))
    return distance


#余弦相似度
def distance_cosine(x1,x2):
    
    vec1 = np.array(x1)
    vec2 = np.array(x2)
    return 1 - np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


