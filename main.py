import json
import utils
import models


if __name__ == '__main__':
    # 生成索引
    print('******************************************')
    print('*    欢迎使用y0ung的中文信息检索系统     *')
    print('*  本系统默认文档库为./archives/Sci-Fi/  *')
    print('* 本系统默认停用词表为./cn_stopwords.txt *')
    print('******************************************')
    generate_index = input('$ 是否需要重新生成索引？(y/n)：')
    if generate_index == 'y':
        utils.docs2index('./archives/Sci-Fi/')
        print('索引生成完毕！')
    else:
        print('未生成新索引！')

    # 读取索引
    with open('docID_index.json', 'r', encoding='utf-8') as f:
        docID_index = json.load(f)
    with open('inverted_index.json', 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)
    print('索引读取完毕！文档库中共有{}个文档，共包含{}个词项'.format(len(docID_index), len(inverted_index)))

    flag = True
    while flag:
        # 获取并预处理输入
        query = input('$ 请输入中文查询内容（输入q退出）：')
        if query == 'q':
            flag = False
            continue
        query = utils.preprocess(query)

        model = input('$ 请选择查询模型（输入1代表布尔模型，2代表向量空间模型，3代表帮助）：')
        # 布尔模型查询
        if model == '1':
            docNames, locations = models.booleanQuery(query, docID_index, inverted_index)
            if len(docNames) != 0:
                print('共找到{}个相关文档：'.format(len(docNames)))
                for i, docName in enumerate(docNames):
                    print('----------------------------------------')
                    print('|第{}个文档：{}'.format(i+1, docName))
                    if len(locations) != 0:
                        print('|行号：{}'.format(locations[i]))
                print('----------------------------------------')
                continue
        # 向量空间模型查询
        elif model == '2':
            docNames_similar, similarities = models.vectorQuery(query, docID_index, inverted_index)
            if similarities[0] != 0:
                print('该查询与文档库的相关性（由大到小）如下：')
                for i, docName in enumerate(docNames_similar):
                    print('----------------------------------------')
                    print('|第{}个文档：{}'.format(i+1, docName))
                    print('|相似度：{}'.format(similarities[i]))
                print('----------------------------------------')
                continue
        # 帮助
        elif model == '3':
            print('1.布尔模型：')
            print('输入查询内容，当所有查询词项在同一文档的同一行时，系统将返回该文档及其所在行号，若仅在同一文档而非同一行，则返回该文档')
            print('2.向量空间模型：')
            print('输入查询内容，系统将根据相关性（基于余弦相似度），由大到小地返回文档库中的所有文档及其相似度')
            continue
        else:
            print('错误：请在 1、2、3 这三个值中选择一个输入！')
            continue

        # 拼写检查
        print('未找到与{}相关的文档，建议更换模型或进行拼写检查！'.format(query))
        spell_check = input('$ 是否需要拼写检查？(y/n)：')
        if spell_check == 'y':
            query_checked = utils.spellCheck(query, inverted_index)
            print('您可能想要查询：{}'.format(query_checked))

    print('感谢使用，再见！')
