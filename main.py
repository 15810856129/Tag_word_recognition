# -*- coding: utf-8 -*-

import os
import gensim
import numpy as np
import matplotlib.pyplot as plt
from text_preprocess import text_preprocess_main
from tag_word_vector import tag_word2vector
from clustering import cluster_main
from sentence_to_vector import buildWordVector
from compute_text_similarity import compute_text_similarity
from tags_performance_test import compute_metrics


def load_word2vec_model():
    '''
        导入训练好的word2vec模型, 避免重复加载模型.
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    model = gensim.models.Word2Vec.load(os.path.join(file_path, 'w2v', 'model'))
    
    return model
    

def incremental_training_word2vec(new_sentences, model):
    '''
        增量式训练word2vec模型.
        在先前训练好的word2vec模型基础上, 使用新的语料继续进行训练.
    '''
    
    model.build_vocab(new_sentences, update=True)
    model.train(new_sentences, total_examples=model.corpus_count, epochs=model.iter)
    
    return model
    

def filter_candidate_tags_by_w2v(model, candidate_tags, tag_library, K=10, threshold=0.8):
    '''
       使用word2vec模型对候选标签词进行过滤，滤除不在word2vec词典中的词.
       对每个候选词计算出的K个关联词在基础标签词字典中进行查询；
       过滤掉其K个关联词有出现在基础标签词字典的候选词.
    '''
    
    # 获得训练好的 word2vec 模型的词库.
    w2v_word_dict = model.wv.vocab
    
    # 对每个候选词计算出的K个关联词在基础标签词字典中进行查询
    new_tags = []
    for keys in candidate_tags:
        sim_list = []
        if keys in w2v_word_dict:
            sim_list.append(model.most_similar(keys, topn=K))
            for i, (key, value) in enumerate(sim_list[0]):
                if key in tag_library and value >= threshold:
                    break
                elif i == len(sim_list[0]) - 1:
                    new_tags.append(keys)
                else:
                    continue   
    return new_tags
    
       
def plot_sim_vs_tags_curve(model, candidate_tags, tag_library):
    '''
        选取相似度阈值，统计候选标签词在全语料中搜索相似词个数和相似度阈值的关系（绘制曲线）;
        以及搜索出的相似词中含基础标签词个数与相似度阈值的关系（绘制曲线）.
    '''
    
    # 获得训练好的 word2vec 模型的词库.
    w2v_word_dict = model.wv.vocab
    
    # 指定相似度阈值范围, 在全语料中计算候选标签词得语义相近词
    similarity = []
    count = []
    for i in range(1, 501, 10):
        temp = []
        for keys in candidate_tags:
            if keys in w2v_word_dict:
                temp.append(model.most_similar(keys, topn=i))
        similarity.append(sum([u[-1][-1] for u in temp]) / len([u[-1][-1] for u in temp]))
        count.append(sum([1 if line[0] in tag_library else 0 for lines in temp for line in lines]) / len([u[-1][-1] for u in temp]))
        
    
    # 绘制曲线图
    plt.plot(similarity, list(range(1, 501, 10)), label='dist vs word numbers', linewidth=3, color='r', marker='o')
    plt.plot(similarity, count, label='dist vs basic tag numbers', linewidth=3, color='b', marker='*')
    plt.xlabel('word numbers')
    plt.ylabel('the threshold of similarity')
    plt.title('The relationship between semantic similarity threshold and tag word size ')
    plt.legend()
    plt.show()

    
def compute_dist_func(A, Mode=0):
    '''
        计算多个点到中心点的平均距离.
        Mode=0,选用欧式距离度量;
        Mode=1,选用余弦距离度量;
        Mode=2,选用L1范数(切比雪夫距离)度量.
    '''
    
    m, n = A.shape
    mean = A.mean(axis=0)
    res = 0
    
    if Mode == 0:
        func = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
    elif Mode == 1:
        func = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * (np.linalg.norm(y)))
    elif Mode == 2:
        func = lambda x, y: np.abs(x - y).max()
    else:
        raise IOError("The input Mode value is illegal, the optional parameter is 0,1,2")
    
    for line in A:
        res += func(line, mean)
    res /= m
    
    return res, func

    
def compute_center_dist(model, new_tags, tag_library, Mode=0):
    '''
        计算两个聚类中心点之间的距离:
    '''
    
    # 计算两个中心点
    try:
        center1 = model[[key for key in new_tags if key in model.wv.vocab]].mean(axis=0)
        center2 = model[[key for key in tag_library if key in model.wv.vocab]].mean(axis=0)
    except:
        raise ZeroDivisionError
        
    # 按指定的距离度量准则进行计算
    res1, func = compute_dist_func(model[[key for key in new_tags if key in model.wv.vocab]], Mode=Mode)
    res2, func = compute_dist_func(model[[key for key in tag_library if key in model.wv.vocab]], Mode=Mode)
    dist = func(center1, center2)

    return dist, res1, res2
    
    
def plot_precision_recall_curve(x, y, xname, yname, z=None):
    
    plt.plot(x, y, color='r', marker='o', label='first line')
    if z != None:
        plt.plot(x, z, c='b', marker='^', label='sencod line')
    plt.xlabel('%s' % xname)
    plt.ylabel('%s' % yname)
    plt.title('the %s - %s curve under different keywords.' % (xname, yname))
    plt.legend()
    plt.show()
    
    
def process_main(K_words=3, K_sentence=5, K=25, threshold=0.95, Mode=1):
    '''
        K_words是候选标签词的词频阈值; K_sentence是训练word2vec句子语料长度的阈值;
        threshold是计算候选标签词的近义词相似度的阈值.
    '''
    
    candidate_tags, clean_word_list, tag_library, texts, labels = text_preprocess_main(K_words=K_words, K_sentence=K_sentence)
    model = load_word2vec_model()
    model = incremental_training_word2vec(clean_word_list, model)
    
    # 获取新的标签词及无监督计算基础标签词和新标签词的距离
    new_tags = filter_candidate_tags_by_w2v(model, candidate_tags, tag_library, K=K, threshold=threshold)
    dist, res1, res2 = compute_center_dist(model, new_tags, tag_library, Mode=Mode)

    new_tags_dict, basic_tags_dict = tag_word2vector(model, tag_library, new_tags)
    plot_sim_vs_tags_curve(model, candidate_tags, tag_library)
    res = cluster_main(basic_tags_dict, n_cluster=30)
    result = cluster_main(new_tags_dict, n_cluster=10)

    
    # 给每条文本预测标签
    tags = [line for line in tag_library] + new_tags
    tags = [key for key in tags if key in model.wv.vocab]
    predict_tags = compute_text_similarity(model, texts, tags)
    
    precision = []
    recall = []
    absolute_usage = []
    relative_usage = []
    n_list = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    for i in n_list:
        value1, value2, accu1, accu2 = compute_metrics(labels, predict_tags, n=i, new_tags=new_tags, tag_library=tag_library)
        precision.append(value1)
        recall.append(value2)
        absolute_usage.append(accu1)
        relative_usage.append(accu2)
        
    plot_precision_recall_curve(recall, precision, xname='recall', yname='precision')
    plot_precision_recall_curve(n_list, absolute_usage, xname='tag number', yname='usage rate', z=relative_usage)
    
    
    return dist, res1, res2, precision, recall, absolute_usage, relative_usage
    
    


if __name__ == '__main__':
    
    dist, res1, res2, precision, recall, absolute_usage, relative_usage = process_main(K_words=3, K_sentence=5, K=10, threshold=0.8)

    