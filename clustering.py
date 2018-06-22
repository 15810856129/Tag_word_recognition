# -*- coding: utf-8 -*-

import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
import sys
sys.setrecursionlimit(5000)


def K_Means_train(x, n_cluster):
    '''
        使用KMeans对标签词进行聚类.
    '''

    # 建立 KMeans 聚类模型
    km = KMeans(n_clusters=n_cluster)
    km.fit(list(x.values()))
    
    # 获取聚类标签
    label_pred = km.labels_
    
    result = dict(zip(x.keys(), label_pred))
    return result
    

def pca_reduction(x):
    
    '''
        使用PCA进行降维处理.
    '''    
    
    keys = []
    values = np.array([0.0]*100).reshape((1, 100))
    
    for key, value in x.items():
        keys.append(key)
        values = np.concatenate((values, value.reshape((1, 100))), axis=0)
        
    print(values.shape)
        
    pca = PCA(n_components=100)
    new_x = pca.fit_transform(values[1:, :], )
    m, n = new_x.shape
    
    
    if n < 100:
        new_x = np.concatenate((new_x, np.zeros((m, 100-n))), axis=1)
    else:
        pass
    
    new_x = dict(zip(keys, new_x))
    
    variance = pca.explained_variance_ratio_
    
    return new_x, variance
    
    
def plot_cluster_picture(new_x, result, n_cluster, tag_word=None):
    '''
        对聚类后的单词进行二维可视化, 只选取前两个维度来绘制.
    '''
    
    n = n_cluster   
    i = len(set(result.values()))
    marks = ['.', 'o', 'v', ',', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    if i <= 0:
        plt.legend()
        plt.show()
        return None
    else:
        temp = []
        keys = []
        
        for key, value in result.items():
            if value == n - i:
                temp.append(new_x[key])
                keys.append(key)
        
        temp = np.array(temp).reshape((len(temp), 100))
        print(temp.shape)
        
        plt.scatter(temp[:, 0], temp[:, 1], c='g')
        plt.xlabel('first component')
        plt.ylabel('sencond component')
        plt.title('Query related tag words visualization results')
        i -= 1
        
        for line in keys:
            new_x.pop(line)
            result.pop(line)
        
        plot_cluster_picture(new_x, result, n_cluster)
        
    
def cluster_main(x, n_cluster):
    
    result = K_Means_train(x, n_cluster=n_cluster)
    new_x, variance = pca_reduction(x)
    plot_cluster_picture(new_x, result.copy(), n_cluster=n_cluster, tag_word=None)
    
    return result
        
  