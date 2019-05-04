# encoding:utf-8
from math import log
import operator


def cal_Entropy(dataset):
    dataset_num = len(dataset)
    label_cnt = {}
    for line in dataset:
        # 得到每行数据的类别
        label = line[-1]
        if label not in label_cnt.keys():
            label_cnt[label] = 0.0
        label_cnt[label] += 1.0
    # 计算熵
    entropy = 0.0
    for key in label_cnt.keys():
        w_i = label_cnt[key]/float(dataset_num)
        entropy -= w_i * log(w_i,2)
    return entropy

def split_dataset(data_set, axis, value):
    '''
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 按某个特征分类后的数据
    '''
    result_dataset = []
    for featVec in data_set:
        if featVec[axis]==value:
            reduced_feat_vec =featVec[:axis]
            reduced_feat_vec.extend(featVec[axis+1:])
            result_dataset.append(reduced_feat_vec)
    return result_dataset

def cal_condition_entropy(data,i,unique_vals):
    # 计算条件熵
    cond_entropy = 0.0
    for feature in unique_vals:
        sub_set = split_dataset(data,i,feature)
        w_i = len(sub_set)/float(len(data))
        cond_entropy += w_i * cal_Entropy(sub_set)
    return cond_entropy

def cal_info_gain(data_set,base_entropy,i):
    # 计算每个特征维度对应的信息增益
    featList = [example[i] for example in data_set]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    new_entropy = cal_condition_entropy(data_set, i, uniqueVals)
    info_gain = base_entropy - new_entropy  # 信息增益，就yes熵的减少，也就yes不确定性的减少
    return info_gain

def choose_best_feature_to_split(data_set):
    # 选择最优的分类特征
    num_features = len(data_set[0]) - 1
    base_entropy = cal_Entropy(data_set)  # 原始的熵
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        info_gain = cal_info_gain(data_set,base_entropy,i)
        if (info_gain > best_info_gain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majorityCnt(classList):
    '''
    采用多数表决的方法决定叶节点属于哪一类
    :param classList: 类标签列表
    :return: 按分类后类别数量排序
    '''
    class_count={}
    # 统计所有类标签的数量
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote]=0
        class_count[vote]+=1
    sortedClassCount = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def create_tree(dataSet, labels):
    '''
    :param dataSet: 训练数据集
    :param labels: 类别标签
    :return: 决策树
    '''
    # 得到类别
    class_list=[example[-1] for example in dataSet]  # 类别：男或女
    # 递归结束条件：所有的类标完全相同
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归结束条件：用完了所有的特征
    if len(dataSet[0]) == 1:
        return majorityCnt(class_list)
    # 选择最优划分特征
    bestFeat = choose_best_feature_to_split(dataSet)
    best_feat_label = labels[bestFeat]
    # 分类结果以字典形式保存
    result_tree = {best_feat_label:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #复制所有的类标签，保证每次递归调用时不改变原始列表的内容
        subLabels=labels[:]
        result_tree[best_feat_label][value]=create_tree(split_dataset\
                            (dataSet,bestFeat,value), subLabels)
    return result_tree

def create_dataset():
    data_set = [['long', 'low', 'man'],
               ['short', 'low', 'man'],
               ['short', 'low', 'man'],
               ['long', 'high', 'woman'],
               ['short', 'high', 'woman'],
               ['short', 'low', 'woman'],
               ['long', 'low', 'woman'],
               ['long', 'low', 'woman']]
    labels = ['hair','voice']
    return data_set,labels

if __name__=='__main__':
    dataSet, labels=create_dataset()
    print(create_tree(dataSet, labels))