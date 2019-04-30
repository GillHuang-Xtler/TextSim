import string

def readFile(train_file):
    with open(train_file,'r') as file_object:
        reader= file_object.read().splitlines()
    # a =''.join(c for c in reader if c not in string.punctuation)
    # print(a)
    readerList=[i for i in reader if i != '']
    readerDic={}
    for i in range(len(readerList)):
        readerDic[i]=readerList[i]
    return readerDic



if __name__=="__main__":
    train_file="./demo.txt"
    print(readFile(train_file=train_file))