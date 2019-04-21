#encoding=utf-8
#朴素贝叶斯用于文档的分类
import numpy as np
# 构建测试数据
def load_data_set():
    posting_list=[
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    class_vec = [0,1,0,1,0,1]
    return posting_list,class_vec

#创建所有文档中包含的词汇表
def create_vecab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        # 将当前文档的单词唯一化然后以并集的方式加入到保存词汇的集合中
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

#将每篇文档转换为词向量
def doc_word2vec(vocablist,input_set):
    return_vec = [0] * len(vocablist)
    for word in input_set:
        if word in vocablist:
            return_vec[vocablist.index(word)] = 1
        else:
            print 'words %s is not in the vocanlist' %word
    return return_vec

# def test():
#     list_posts,list_class = load_data_set()
#     my_vocablist = create_vecab_list(list_posts)
#     print doc_word2vec(my_vocablist,list_posts[0])
# test()

# 通过贝叶斯算法对数据集进行训练，从而统计出所有词向量分类的概率
# 将待分类的文档转变为词向量以后，从训练集中取出该词向量的概率，概率最大的对应就是其分类结果

#计算某个词向量的概率
def trainNB(train_matrix,train_category):
    # 文档个数
    num_tiain_docs = len(train_matrix)
    # 文档词数
    num_words = len(train_matrix[0])
    # 文档属于分类1的概率
    prob_of_doc = sum(train_category)/float(num_tiain_docs)
    # 属于分类0的词向量求和
    # p0num = np.zeros(num_words)
    # p1num = np.zeros(num_words)
    #tips：避免其中的一个累计值为0，导致整体结果为0
    p0num = np.ones(num_words)
    p1num = np.ones(num_words)

    #分类 0/1 的所有文档内所有单词数统计
    # p1_denom = 0.0
    # p0_denom = 0.0
    p1_denom = 2.0
    p0_denom = 2.0
    for i in range(num_tiain_docs):
        if train_category[i] == 1:
            #词向量累加
            p1num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 各单词在分类0/1 下的概率
    # p1vect = p1num/p1_denom
    # p0vect = p0num/p0_denom
    # 避免数值过小，产生精度下溢的问题
    p0vect = np.log(p0num / p0_denom)
    p1vect = np.log(p1num / p1_denom)

    return p0vect,p1vect,prob_of_doc

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    # 为分类1的概率
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    # 为分类0的概率
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':

    list_posts, listClasses = load_data_set()
    myVocabList = create_vecab_list(list_posts)

    # 由训练文档得到词向量矩阵
    train_matrix = []
    for list_doc in list_posts:
        train_matrix.append(doc_word2vec(myVocabList, list_doc))

    # 训练词向量矩阵得到词语对应类别的出现概率，以及文章类别概率
    p0vec, p1vec, p1 = trainNB(np.array(train_matrix), np.array(listClasses))

    # 测试一
    test1 = ['love', 'my', 'dog']
    thisDoc = np.array(doc_word2vec(myVocabList, test1))
    print test1, classifyNB(thisDoc, p0vec, p1vec, p1)

    # 测试二
    test2 = ['stupid', 'problem']
    thisDoc = np.array(doc_word2vec(myVocabList, test2))
    print test2, classifyNB(thisDoc, p0vec, p1vec, p1)