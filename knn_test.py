# coding:utf-8
# http://www.voidcn.com/article/p-ukvqnrzd-qq.html
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

# 查看iris数据集
iris = load_iris()
print iris

knn = neighbors.KNeighborsClassifier()
# 训练数据集
knn.fit(iris.data, iris.target)
# 预测
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print predict
print iris.target_names[predict]