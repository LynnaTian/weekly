# encoding:utf-8
import jieba
from sklearn .feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
# seg_list = jieba.cut("我来到北京清华大学")
# print ','.join(seg_list)

file = open('data.txt')
file_list = file.read().split('\n')
data = []
for line in file_list:
    data.append(' '.join((jieba.cut(line))))

# method1:sklearn
# 将得到的词语转换为词频矩阵
vectora = CountVectorizer()
X =vectora.fit_transform(data)

# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
weight = tfidf.toarray()

for i in range(len(weight)):
    print weight[i]
words = vectora.get_feature_names()
for i in words:
    print i

# method2:defined

# 词在文档中出现的个数
cfs = []
cfs.extend([float(e.count(word)) for word in words] for e in data)
cfs = np.array(cfs)
print 'cfs :\n',cfs

# cfs2 = []
# for e in data:
#     cf = [float(e.count(word)) for word in words]
#     cfs2.append(cf)
# print cfs2

#词在文档中出现的频率
# tfs = []
# for e in cfs2:
#     tf = e/(np.sum(e))
#     tfs.append(tf)
# print 'tfs:\n',np.array(tfs)

tfs2 = []
tfs2.extend(e/(np.sum(e)) for e in cfs)
tfs2 = np.array(tfs2)
print 'tfs:\n',tfs2

#含词的文档数
# dfs = list(np.zeros(len(words),dtype=int))
# for i in range(len(words)):
#     for term in data:
#         if term.find(words[i]) != -1:
#             dfs[i] += 1
# print dfs

dfs2 = []
for i in range(len(words)):
    onehot = [(item.find(words[i]) != -1 and 1 or 0) for item in data]
    dfs2.append(onehot.count(1))
print 'dfs\n',dfs2

#计算idf
#log10(N/(1+DF))
N = np.shape(data)[0]
idfs = []
for e in dfs2:
    idfs.append((np.log10(N*1.0/(1+e))))
print 'idf\n',np.array(idfs)

#计算 tf-idf
tfidfs = []
for i in range(np.shape(data)[0]):
    word_tfidf = np.multiply(tfs2[i],idfs)
    tfidfs.append(word_tfidf)
print 'tfdifs:\n',np.array(tfidfs)