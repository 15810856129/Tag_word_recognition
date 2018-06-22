# -*- coding: utf-8 -*-
import os
import numpy as np
import collections


def buildWordVector(model, words, size=100):
    '''
        对每个句子的所有词向量取加权均值来作为此条句子的句向量.
        size是词向量的维度,权重采用单词的词频tf.
    '''
    
    # 获取词频 tf
    tf = collections.Counter(words)
    tf = {key: value / max(tf.values()) for key, value in tf.items()}
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    
    for word in enumerate(words):
        try:
            vec += tf[word] * model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
        
    if count != 0:
        vec /= count
        
    return vec
