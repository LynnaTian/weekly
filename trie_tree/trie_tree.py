# encoding:utf-8
class Node():
    def __init__(self):
        self.childs = [None] * 26
        self.isLeaf = False


class Trie(object):
    def __init__(self):
    # 建立节点
        self.root = Node()

    def insert(self, word):
    # 插入单词
        self.inserthelper(word, self.root)

    def inserthelper(self, word, node):
        if node == Node:
            return
        if len(word) == 0:
            node.isLeaf = True
            return
        index = ord(word[0]) - ord('a')
        if node.childs[index] is None:
            node.childs[index] = Node()
        self.inserthelper(word[1:], node.childs[index])

    def search(self, word):
    # 查询
        return self.searchhepler(word, self.root)

    def searchhepler(self, word, node):
        if node is None:
            return False
        if len(word) == 0:
            return node.isLeaf
        index = ord(word[0]) - ord('a')
        return self.searchhepler(word[1:], node.childs[index])

    def startsWith(self, prefix):
    # 前缀查询
        return self.startsWithhelper(prefix, self.root)

    def startsWithhelper(self, prefix, node):
        if node is None:
            return False
        if len(prefix) == 0:
            return True
        index = ord(prefix[0]) - ord('a')
        return self.startsWithhelper(prefix[1:], node.childs[index])


if __name__ == '__main__':
    trie = Trie()
    trie.insert("apple")
    print(trie.search("apple"))
    print(trie.search("app"))
    print(trie.startsWith("app"))
    trie.insert("app")
    print(trie.search("app"))
