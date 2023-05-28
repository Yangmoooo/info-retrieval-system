import numpy as np
import utils
from sympy import S

# 布尔模型查询
def booleanQuery(words, docID_index, inverted_index):
    # 获取文档ID交集
    docIDs = S.UniversalSet
    for word in words:
        if word in inverted_index:
            posting_list = inverted_index[word]['IDs']
            docIDs &= set(posting_list)
    if docIDs == S.UniversalSet or len(docIDs) == 0:
        return [], []
    docIDs = [str(docID) for docID in docIDs]
    # 获取查询词项在同一文档、同一行的文档名和位置
    docNames = list()
    locations = list()
    for docID in docIDs:
        location = S.UniversalSet
        for word in words:
            if word in inverted_index:
                pos = inverted_index[word]['IDs'][docID]
                location &= set(pos)
        if len(location) != 0:
            docNames.append(docID_index[docID]['name'])
            locations.append(sorted(location))
    # 查询词项是否在同一行
    if len(docNames) != 0:
        return docNames, locations
    else:
        return [docID_index[docID]['name'] for docID in docIDs], []

# 向量空间模型查询
def vectorQuery(words, docID_index, inverted_index):
    # 计算查询向量
    query_vec = list()
    for word in inverted_index:
        if word in words:
            query_vec.append(utils.TFIDF(word, words, docID_index, inverted_index))
        else:
            query_vec.append(0)
    query_vec = np.array(query_vec)
    # 计算各文档向量
    doc_vecs = np.zeros((len(docID_index), len(inverted_index)))
    for i, docID in enumerate(docID_index):
        for j, word in enumerate(inverted_index):
            if docID in inverted_index[word]['IDs']:
                doc_vecs[i, j] = utils.TFIDF(word, docID, docID_index, inverted_index)
    # 计算向量间的余弦相似度
    similarities = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
    # 相似度从大到小排序
    docID_similar = np.argsort(-similarities)
    similarities = list(similarities[docID_similar])
    docNames_similar = list()
    for docID in docID_similar:
        docNames_similar.append(docID_index[str(docID)]['name'])
    # 返回排序后的文档名和相似度
    return docNames_similar, similarities