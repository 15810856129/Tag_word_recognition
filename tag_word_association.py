# -*- coding: utf-8 -*-
from process import load_word2vec_model
from process import incremental_training_word2vec



def associated_tag_word_query(tag_word, model, K=10):
    '''
        给定一个词语, 在语料库中查询与其关联的K个标签词;
        其结果是返回一个顺序标签词列表，排序规则是每个标签词与查询词的语义相似度较大，
        并且当前位置的标签词与前一个标签词差异较大. (筛选出的标签词是‘好而不同’的.)
        
        Equation：min f(t) = [cos(V(t-1), V(t))] / [cos(V(0), V(t))] , (1 < t < K)
        其中, t是第t个要选取的标签词, N是总共要筛选出的标签词数目.
    '''
    
    if tag_word in model.wv.vocab:
        sim_list = model.most_similar(tag_word, topn=K)
        similar_list = sim_list.copy()[1:]
        
        sequence_list = []
        sequence_list.append(sim_list[0])
        
        try:
            for i in range(1, len(sim_list)):
                pointer = 0
                temp = model.similarity(similar_list[0][0], sequence_list[0][0]) / model.similarity(tag_word, similar_list[0][0])
                
                for j, (key, value) in enumerate(similar_list[1:]):
                    ans = model.similarity(key, sequence_list[i-1][0]) / model.similarity(tag_word, key)
                    
                    if ans >= temp:
                        temp = ans
                        pointer = j + 1
                    else:
                        continue
                    
                if pointer > 0:
                    sequence_list.append(similar_list.pop(pointer))
                else:
                    sequence_list.append(similar_list.pop(0))  
        except:
            raise ZeroDivisionError('The denominator in the starting operation is 0!')
    else:
        raise IOError('the input word is not in word2vec vocabulary !')
    
    # 获取查询到的关联标签词的词向量
    sequence_dict = {key: model[key] for (key, _) in sequence_list}
    sequence_dict.update({tag_word: model[tag_word]})
    
    return sequence_list, sim_list, sequence_dict
 
    
    
def cluster_visualization(sequence_dict, n_cluster=5):
    '''
        对查询到的关联标签词进行聚类可视化展示
    '''
    # 对查询到的已排序的关联标签词进行聚类
    cluster_main(sequence_dict, n_cluster)
    

    

if __name__ == '__main__':
    
    model = load_word2vec_model()
    new_sentences = [['你', '真的', '是', '一个', '好人']]   
    model = incremental_training_word2vec(new_sentences, model)
    sequence_list, origin_sim_list, sequence_dict = associated_tag_word_query(tag_word='幸福', model=model, K=20)
    cluster_visualization(sequence_dict)
    
