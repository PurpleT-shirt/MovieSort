import numpy as np
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    label = ['A','A','A','B','B','B']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('X')
    plt.ylabel('Y')
    #ax1.scatter(x, y, c='r', marker='o')
    ax.scatter(group[:,0],group[:,1])
    plt.show()
    #return group,label

def classifier(inX, dataSet, labels, k):
    #数据集大小
    dataSetSize = dataSet.shape[0]

    #计算距离
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2 #2次方
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5

    #距离排序
    sortedDistances = distance.argsort()

    #统计前k个点所属的类别
    classCount = {}
    for i in range(k):
        votaIlabel = labels[sortedDistances[i]]
        classCount[votaIlabel] = classCount.get(votaIlabel,0) + 1      #dict.get(key, default=None),key字典中要查找的键;default -- 如果指定键的值不存在时，返回该默认值值
    SortedClassCount = sorted(classCount.items(), key = lambda x:x[1], reverse=True)  #reverse,逆序
    return SortedClassCount[0][0]

if __name__ == '__main__':
    #group, label = createDataSet()
    #print(classifier([0,100], group, label, 3))
    createDataSet()