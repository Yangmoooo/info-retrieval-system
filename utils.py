import jieba
import pypinyin
import utils
import math
import os
import json
from collections import Counter


# 读取文档并建立索引
def docs2index(dir_path):
    # 初始化文档ID索引和倒排索引
    docID = -1
    docID_index = dict()
    inverted_index = dict()
    # 读取文档并建立索引
    for root, dir, files in os.walk(dir_path):
        for file in files:
            # 分词
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                continue
            words = jieba.lcut(text)
            # 获取每个词所在的原始行号
            row = 1
            for i, word in enumerate(words):
                words[i] = [word, row]
                if word == '\n':
                    row += 1
            # 语言预处理
            words = utils.preprocess(words)
            # 建立文档ID索引
            docID += 1
            docID_index[docID] = dict()
            docID_index[docID]['name'] = os.path.join(root, file)
            docID_index[docID]['length'] = len(words)
            # 建立倒排索引
            for word in words:
                if word[0] not in inverted_index:
                    inverted_index[word[0]] = dict()
                    inverted_index[word[0]]['IDs'] = dict()
                    inverted_index[word[0]]['freq'] = 0
                if docID not in inverted_index[word[0]]['IDs']:
                    inverted_index[word[0]]['IDs'][docID] = list()
                inverted_index[word[0]]['IDs'][docID].append(word[1])
                inverted_index[word[0]]['freq'] += 1
    # 保存索引
    with open('docID_index.json', 'w', encoding='utf-8') as f:
        json.dump(docID_index, f, ensure_ascii=False)
    with open('inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False)

# 拼写检查
def spellCheck(query, inverted_index):
    # 构建词库拼音索引
    pinyin_index = dict()
    for word in inverted_index:
        word_pinyin = ''.join(pypinyin.lazy_pinyin(word))
        if word_pinyin not in pinyin_index:
            pinyin_index[word_pinyin] = list()
            pinyin_index[word_pinyin].append(word)
        elif len(pinyin_index[word_pinyin]) == 1:
            if inverted_index[word]['freq'] > inverted_index[pinyin_index[word_pinyin][0]]['freq']:
                pinyin_index[word_pinyin].insert(0, word)
            else:
                pinyin_index[word_pinyin].append(word)
        else:
            for i, term in enumerate(pinyin_index[word_pinyin]):
                if inverted_index[word]['freq'] > inverted_index[term]['freq']:
                    pinyin_index[word_pinyin].insert(i, word)
                    break
    # 拼音纠错
    query_checked = list()
    for word in query:
        word_pinyin = ''.join(pypinyin.lazy_pinyin(word))
        if word_pinyin in pinyin_index:
            if word != pinyin_index[word_pinyin][0]:
                query_checked.append(pinyin_index[word_pinyin][0])
            else:
                query_checked.append(pinyin_index[word_pinyin][1])
    # 返回校正后的查询词项列表
    return query_checked

# 读取停用词表
def getStopWords(stop_words_path):
    stop_words = set()
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    return stop_words

# 语言预处理
def preprocess(words):
    stop_words = utils.getStopWords('cn_stopwords.txt')
    if isinstance(words, str):
        # 处理用户查询语句
        words = jieba.lcut(words)
        words = [word for word in words if '\u4e00' <= word <= '\u9fa5' and word not in stop_words]
    elif isinstance(words, list):
        # 处理带行号的词表
        words = [word for word in words if '\u4e00' <= word[0] <= '\u9fa5' and word[0] not in stop_words]
    else:
        raise TypeError("words must be str or list")
    return words

# 计算TF-IDF
def TFIDF(word, doc, docID_index, inverted_index):
    # 计算词频TF
    if type(doc) == list:
        # 计算查询向量TF，此时doc是分词处理后的查询词项列表
        words_cnt = Counter(doc)
        tf = words_cnt[word] / len(doc)
    elif type(doc) == str:
        # 计算文档向量TF，此时doc是文档ID
        doc_word_cnt = len(inverted_index[word]['IDs'][doc])
        doc_allwords_cnt = docID_index[doc]['length']
        tf = doc_word_cnt / doc_allwords_cnt
    else:
        raise TypeError("doc must be list or str")
    # 计算逆文档频率IDF
    idf = math.log(len(docID_index) / len(inverted_index[word]['IDs'])) + 1
    # 返回TF-IDF
    return tf * idf
