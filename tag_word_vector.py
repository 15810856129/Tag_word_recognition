# -*- coding: utf-8 -*-

def tag_word2vector(model, tag_library, new_tags):
    '''
        导入word2vec模型,并过滤掉不在word2vec模型中出现的词, 以及获得每个词的词向量表示.
    '''
    
    w2v_word_dict = model.wv.vocab
    new_tags_dict = {key: model[key] for key in new_tags if key in w2v_word_dict.keys()}
    basic_tags_dict = {key: model[key] for key in tag_library if key in w2v_word_dict.keys()} 

    return new_tags_dict, basic_tags_dict