
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
import numpy as np


def readFile(train_file):

    with open(train_file,'r') as file_object:
        reader= file_object.read().splitlines()
    # a =''.join(c for c in reader if c not in string.punctuation)
    # print(a)
    readerList=([i for i in reader if i != ''])
    readerList=list(readerList)
    readerDic={}
    for i in range(len(readerList)):
        readerDic[i]=readerList[i]
    return readerList,readerDic

def tdIdf(readerList):

    # readerList = ["I have a bad pen.","I have an apple.","I eat an apple",]
    tf = TfidfVectorizer(ngram_range=(1,2),analyzer='word',smooth_idf=1) # 按照 word 来做特征，最大范围是两个单词，最小是一个单词
    discuss_tf = tf.fit_transform(readerList)
    print("----------每个短文本生成的TDIDF向量为：------------")
    print(discuss_tf.todense())
    # words = tf.get_feature_names()
    # print(words)
    # print(discuss_tf)
    # num = len(words)
    # row_num=len(readerList)
    # for i in range(row_num):
    #     print('----Document %d----'%(i))
    #     for j in range(num):
    #         print(words[j],discuss_tf[i,j])
    return discuss_tf.todense()

def cos(vector1,vector2):

    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)

def cosSimilarity(x, y, norm=False):

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos

def cosMatrix(vecs):
    cos_matrix=np.zeros(shape=(len(vecs),len(vecs)))
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            cos_matrix[i][j]=cosSimilarity(vecs[i],vecs[j])
    return cos_matrix

def allReduceVector(vecs):
    arvec=[]
    for i in range(len(vecs)):
        avg=sum(vecs[i])/len(vecs[0])
        arvec.append(avg)
    return arvec

def vecPartition(vec):
    left=0;
    partion=[]
    for i in range(1,len(vec)):
        if vec[i]-vec[left]<0.002:
            partion.append(left)
        else:
            left+=1
            partion.append(left)
    return partion

if __name__=="__main__":
    train_file="./demo.txt"
    readerList,readerDic=readFile(train_file=train_file)
    vecs=tdIdf(readerList).getA()
    print("----------两两短文本之间的相似度矩阵为：------------")
    print(cosMatrix(vecs=vecs))
    print("---------------短文本之间的评分为：-----------")
    print(allReduceVector(vecs=vecs))
    print("----------根据均值评分来分类的结果为：----------------")
    print(vecPartition(allReduceVector(vecs)))