# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist 
from sentence_to_vector import buildWordVector

def cosine(A, B):
    '''
        计算向量A和向量B的余弦距离.
    '''
#    dist = np.dot(A, B) / (np.linalg.norm(A) * (np.linalg.norm(B)))
    dist = pdist(np.vstack([A, B]),'cosine') 
    
    return dist
    

def compute_text_similarity(model, texts, tags):
    '''
        给定文本，计算与该文本相似的n个词向量.
    '''
    
    # 获取所有文本的词向量表示
    text_vector = np.zeros((1, 100))

    for value in texts:
        text_vector = np.concatenate((text_vector, buildWordVector(model, value)), axis=0)
    text_vector = text_vector.reshape((len(texts)+1, 100))[1:]

    # 计算与每条文本语义相近的n个词.
    index = []
    for line in text_vector:
        temp = []
        for value in tags:
            temp.append(cosine(model[value], line))
        index.append(temp)
    
    # 按相似度大小进行排序，获取排好序的标签词.
    predict_tags = []
    for line in index:
        predict_tags.append([tags[i] for i, _ in sorted(enumerate(line), key=lambda x: x[1])])
    
    return predict_tags
    
#    sim_list = predict_tags
#    sequence_test_tags = []
#    for i, lines in enumerate(sim_list):
#        similar_list = lines.copy()[1:]
#        sequence_list = []
#        sequence_list.append(lines[0])
#
##        try:
#        for j in range(1, len(lines)):
#            pointer = 0
#            temp = model.similarity(similar_list[0], sequence_list[0]) / cosine(text_vector[i], model[similar_list[0]].reshape(1, 100))
#            
#            for k, key in enumerate(similar_list[1:]):
#                ans = model.similarity(key, sequence_list[j-1]) / cosine(text_vector[i], model[key].reshape(1, 100))
#                
#                if ans >= temp:
#                    temp = ans
#                    pointer = k + 1
#                else:
#                    continue
#                
#            if pointer > 0:
#                sequence_list.append(similar_list.pop(pointer))
#            else:
#                sequence_list.append(similar_list.pop(0))  
#        
#        sequence_test_tags.append(sequence_list)
#            
##        except:
##            raise ZeroDivisionError('The denominator in the starting operation is 0!')
#     
#    return sequence_test_tags, predict_tags
