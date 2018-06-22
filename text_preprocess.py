# coding: utf-8
from __future__ import unicode_literals
import jieba
import yaml
import os


'''
  Function:
 （1）首先，获取人工标注的基础标签词集，并从爬取的微博语料中提取每个博主的标签词和相关评论；
 （2）其次，用结巴分词工具对中文评论语料进行分词；
 （3）然后，去除分词结果中的停用词，过滤掉已在基础标签词集中出现的博主的标签词，并再次过滤掉在
     语料中出现频次较低的博主的标签词后作为候选标签词集。
'''


def load_user_library(file):
    '''
        Load user dictionary to increase segmentation accuracy
    '''
    
    if isinstance(file, str):
        jieba.load_userdict(file)
    elif isinstance(file, list):
        for value in file:
            jieba.add_word(value.lower())
    else:
        pass
    

def load_conf_file():
    '''
        as a tool to load conf file 
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__)) 
    with open(file_path + '/conf' + '/stop_words.yaml', 'r') as f1:
        stopwords = set(yaml.load(f1))
    
    tag_library = {}
    with open(file_path + '/conf' + '/tag_library_v1.0_20170616', 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
             tag_library[line.strip('\n')] = 1
    
    return stopwords, tag_library


def segmentation(sentence, para='list'):
    '''
        use jieba tool to cut sentence
    '''
    
    if para == 'str':
        seg_list = jieba.cut(sentence)
        seg_result = ' '.join(seg_list)
        return seg_result
    
    elif para == 'list':
        seg_result = jieba.lcut(sentence)
        return seg_result
        

def sentence_filter_stopwords(word_list, stopwords):
    '''
        filter stop words when the sentence has been cutted and get the clear words
        clean_word_list: [[], [], []], 嵌套列表，每一个内层列表存放了一条经过处理后的评论.
    '''
    
    # filter stopwords
    clean_word_list = []
    for word in word_list:
        if word in stopwords:
            continue
        else:
            clean_word_list.append(word)       

    return clean_word_list	
    


def extract_user_comment(file_path, tag_library, stopwords, K_words=1, K_sentence=5):
    '''
        从爬取的微博用户信息中提取标签词和该博主的描述文本.
    '''
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.split('\t'))
    
    # 获取语料
    corpus = [v[5] for v in data if v[5] != 'None' and v[5] != '']

    # tag_label是每条文本的标签 [[],[]]
    tag_label = [w[3].split(',') for w in data]
    tag_words = [u for w in data for u in w[3].split(',') if u != 'None']
    
    # 添加新词到 jieba词典
    load_user_library(tag_words)
    
    # 分词
    comment_words = list(map(segmentation, corpus))
    
    # clean_word_list是 [[],[]] 形式，每个内嵌列表代表一个博主的描述文本，用于word2vec训练.
    clean_word_list = list(map(sentence_filter_stopwords, comment_words, len(comment_words) * [stopwords]))
    label_list = [tag_label[i] for i, line in enumerate(clean_word_list) if len(line) >= K_sentence]
    
    clean_word_list = [line for line in clean_word_list if len(line) >= K_sentence]

    # 筛选出评价新标签词时所需的文本--标签词对
    labels = [line for line in label_list if line != ['None']]
    texts = [clean_word_list[i] for i, line in enumerate(label_list) if line != ['None']]
    
    clean_words = [word for words in clean_word_list for word in words]
    
    # 获取博主的标签词
    tag_dict = {key: clean_words.count(key) for key in tag_words}
    
    # 选择阈值过滤掉词频较低并且已经在基础标签词集中的标签词
    candidate_tags = [key for key, value in tag_dict.items() if value >= K_words and key not in tag_library]

    return candidate_tags, clean_word_list, texts, labels
    
    
def text_preprocess_main(K_words=1, K_sentence=5):
    '''
        文本预处理的主函数.
    '''
    
    stopwords, tag_library = load_conf_file()
    
    load_user_library(stopwords)
    load_user_library(tag_library)
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'weibo_account_info_v1.2_20180606')
    
    candidate_tags, clean_word_list, texts, labels = extract_user_comment(file_path, tag_library, stopwords, K_words=K_words, K_sentence=K_sentence)
    
    return candidate_tags, clean_word_list, tag_library, texts, labels
    
    
        
if __name__ == '__main__':

    candidate_tags, clean_word_list, tag_library, texts, labels = text_preprocess_main(K_words=1, K_sentence=5)

    