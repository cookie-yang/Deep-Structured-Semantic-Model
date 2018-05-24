import numpy as np
import tensorflow as tf


WRITEFILEPATH = "../data/wordVector"



def constructSet(StringList,gramSet):
    ENGLISHTAG = 'E'
    ENGLISHFLAG = False
    NUMTAG = 'N'
    NUMFLAG = False
    for e in StringList:
        for c in e:
            if ord(c)>=65 and ord(c) <=122:
                if ENGLISHFLAG == False:
                    gramSet.add(ENGLISHTAG)
                    ENGLISHFLAG = True
                else:
                    continue
            elif ord(c)>=48 and ord(c)<=57:
                if NUMFLAG == False:
                    gramSet.add(NUMTAG)
                    NUMFLAG == True
                else:
                    continue
            else: 
                gramSet.add(c)


def strL2vector(string,vectorDict_p,indexDict_p,len):
    vectorLen = len
    vectorDict = vectorDict_p.copy()
    indexDict = indexDict_p.copy()
    ENGLISHTAG = 'E'
    NUMTAG = 'N'
    
    for e in string:
        if ord(e)>=65 and ord(e) <=122:
            if vectorDict.get(ENGLISHTAG) == None:
                raise Exception("unreasonably initialize vectorDict to include:",e)
            else:
                vectorDict[ENGLISHTAG] +=1
        elif ord(e)>=48 and ord(e)<=57:
            if vectorDict.get(NUMTAG) == None:
                raise Exception("unreasonably initialize vectorDict to include:",e)
            else:
                vectorDict[NUMTAG] +=1
        else: 
            if vectorDict.get(e) == None:
                raise Exception("unreasonably initialize vectorDict to include:",e)
            else:
                vectorDict[e] +=1
    else:
        vector = np.zeros(vectorLen,dtype=np.float32)
        for k,v in vectorDict.items():
            if v != 0:
                vector[indexDict[k]]=v
        else:
            return vector
def wordhash(FILEPATH):
    gramSet = set()
    num = 0
    vectorDict = dict()
    indexDict=dict()
    #line format (id####10*string####20*string####80*string)
    with open(FILEPATH) as file:
        for line in file:
            num +=1
            if num > 2000000:
               break
            line = line.rstrip("\n")
            id,query,positive,negative = line.split("####")
            queryL = list(map(lambda x: x.split(" ")[-1],query.split("\t")))
            positiveL = list(map(lambda x: x.split(" ")[-1],positive.split("\t")))
           
            negativeL = list(map(lambda x: x.split(" ")[-1],negative.split("\t")))
            strList = queryL+ positiveL + negativeL
            constructSet(strList,gramSet)
            print(num,len(gramSet))
            
        print("finish construct set")
        file.close()
    
    for e in gramSet:
        vectorDict[e]=0
    for i,e in enumerate(gramSet):
        indexDict[e] = i
    writer = tf.python_io.TFRecordWriter("../data/data.tf")
    with open(FILEPATH) as file:
        num = 0 
        for line in file:
            num+=1
            if num > 2000000:
               break
            line = line.rstrip("\n")
            id,query,positive,negative = line.split("####")
            queryL = list(map(lambda x: x.split(" ")[-1],query.split("\t")))
            positiveL = list(map(lambda x: x.split(" ")[-1],positive.split("\t")))
            negativeL = list(map(lambda x: x.split(" ")[-1],negative.split("\t")))
            vector_q = strL2vector("".join(queryL),vectorDict,indexDict,len(gramSet))
            vector_p = strL2vector("".join(positiveL),vectorDict,indexDict,len(gramSet))
            stride = len(negativeL)//4
            negativeL = [negativeL[0:stride],negativeL[stride:stride*2],negativeL[stride*2:stride*3],negativeL[stride*3:]]
            vector_n_L = list(map(lambda s:strL2vector("".join(s),vectorDict,indexDict,len(gramSet)),negativeL))
            result = np.array([vector_q,vector_p]+vector_n_L,dtype=np.float32)
            # example = tf.train.Example(features=tf.train.Features(feature={
            #     'q':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[0]])),
            #     'p':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[1]])),
            #     'n_1':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[2]])),
            #     'n_2':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[3]])),
            #     'n_3':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[4]])),
            #     'n_4':tf.train.Feature(float_list=tf.train.FloatList(value=[x for x in result[5]])),
            #     'dim_n':tf.train.Feature(int64_list=tf.train.Int64List(value=[len(gramSet)]))
            # }))
            example = tf.train.Example(features=tf.train.Features(feature={
                'q':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[0].tostring()])),
                'p':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[1].tostring()])),
                'n_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[2].tostring()])),
                'n_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[3].tostring()])),
                'n_3':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[4].tostring()])),
                'n_4':tf.train.Feature(bytes_list=tf.train.BytesList(value=[result[5].tostring()])),
                'dim_n':tf.train.Feature(int64_list=tf.train.Int64List(value=[len(gramSet)]))
            }))
            writer.write(example.SerializeToString())
                            
            print(num)
     
        file.close()
        writer.close()


if __name__ == "__main__":
    wordhash("../data/data")