# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import math


def test_new_tag_words_performance(sentence, n):
    '''
        用TF-IDF和TextRank两种方法给给每个微博用户的描述贴上标签词；
    '''
    
    # 采用 TF-IDF来给每个微博用户的描述贴上标签词.
    key_words1 = jieba.analyse.extract_tags(sentence, topK=n, withWeight=True, 
                                           allowPOS=('ns', 'n', 'vn', 'v'), withFlag=True)
    
    # 采用 TextRank 来给每个微博用户的描述贴上标签词.
    key_words2 = jieba.analyse.textrank(sentence, topK=n, withWeight=True, 
                                        allowPOS=('ns', 'n', 'vn', 'v'), withFlag=True) 
    
    return key_words1, key_words2
    
    
def compute_metrics(labels, predict_tags, n, new_tags, tag_library):
    '''
        计算准确率和召回率
    '''
    count = []
    tp = 0
    for i, lines in enumerate(predict_tags):        
        for line in lines[:n]:
            if line in set(labels[i]):
                tp += 1
                count.append(line)
            else:
                continue
    
    precision = tp / (len(labels))
    recall = tp / (sum([len(set(line)) for line in labels]))
    relative_usage = 100 * len(set(count)) / (len(set(new_tags)))
    absolute_usage = 100 * len(set(count)) / (len(set(tag_library)) + len(set(new_tags)))
    
    return precision, recall, absolute_usage, relative_usage
    