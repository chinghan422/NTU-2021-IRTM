#!/usr/bin/env python
# coding: utf-8

import re
import os
import nltk
import math
import numpy as np
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


def preprocess(text):  # pa1: preprocess function

    # tokenization
    tokens = re.findall("[a-zA-Z]+", text)
    # lower-casing
    lowercase = [token.lower() for token in tokens]
    # stop-words removal
    stops = stopwords.words('english')
    filtered = [w for w in lowercase if w not in stops]
    # stemming
    ps = PorterStemmer()
    stemming = [ps.stem(i) for i in filtered]

    # remove single character like 'a', 'x', 'w'...
    word = [i for i in stemming if len(i) > 1]

    return word


# Training

## Prepare dictionaries of all training data and each class


def getClassesList(C):
    '''
    create dictionary storing Classes List like:
    {1:[11,19,29,113,115,..],
     2:[1,2,3,4,5,6,7,8,9,...],
     ...}

    which can be extracted by:
    classesList[1][0]  # value=11
    classesList[2][4]  # value=5
    classesList[classIndex][orderOfDoc]
    '''
    path = C
    trainDataList = pd.read_csv('./'+ path,
                                delim_whitespace=True,  # list seperated by space
                                header=None,  # data start from first row
                                index_col=0)  # set head of each row to index
    
    classesList={}

    for i in range(1, trainDataList.shape[0]+1):  # trainDataList.shape[0]=13
        # print(i)
        classesList[i] = list()  # initial classX's  dictionary value to list type
        for j in range(1, trainDataList.shape[1]+1):  # trainDataList.shape[1]=15
            # print(j)
            classesList[i].append(trainDataList[j][i])
    
    return trainDataList, classesList


df_trainList, classesList = getClassesList("training.txt")  # get training data

# convert training data to 1-d list
trainList = df_trainList.values.tolist()  # dataframe to list
trainListflat = [x for row in trainList for x in row]  # flatten training data list


# get dictionary from training data
'''
output: 
trainV{(word: document frequency)}
'''
# get training data files list
filenames = []  # store training data files list like:['11.txt','19.txt','29.txt','113.txt',...]

for i in range(len(trainListflat)):
    filenames.append(str(trainListflat[i])+'.txt')

# store document as
# [['apple', 'banana'], ['cat', 'dog']]
documents = []

# {term: doc_fre} e.g. {'apple':2, 'banana':1}
trainV = {}

# {term: index} e.g. {'apple': 41...}
key2Index = {}

# record txt file read sequence
read_sequence = []

for filename in filenames: # traverse files in folder
    read_sequence.append(filename.split(".")[0])
    with open('../pa2/data/'+ filename, 'rb') as txt:  # read txt files
        content = txt.read().decode('latin-1').replace('\n', '')  # decoding and ignore new line
        tokens = preprocess(content)
        documents.append(tokens)
# print(read_sequence)

for document in documents:
    for token in set(document): # counting document frequency
        # print(token)
        if token in trainV:  # if token exists in dictionary
            trainV[token] += 1  # then counter plus one
        else:
            trainV.update({token: 1})  # if token didn't appear in dict., update count to one

len(trainV)  # 5191


# get dictionary from each class
def getDictionary(c):
    
    classTxt = []  # get txt files each class
    for i in range(len(classesList[c])):
        # print(str(classesList[c][i])+'.txt')
        classTxt.append(str(classesList[c][i])+'.txt')

    # store document as
    # [['apple', 'banana'], ['cat', 'dog']]
    documents = []

    # {term: doc_fre} EX: {'apple':2, 'banana':1}
    dict_class = {}

    # {term: index} EX: {'apple': 41...}
    key2Index = {}

    # record txt file read sequence
    read_sequence = []

    for filename in classTxt: # traverse files in folder
        read_sequence.append(filename.split(".")[0])
        with open('../pa2/data/'+ filename, 'rb') as txt:  # read txt files
            content = txt.read().decode('latin-1').replace('\n', '')  # decoding and ignore new line
            tokens = preprocess(content)
            documents.append(tokens)
    # print(read_sequence)

    for document in documents:
        for token in set(document): # counting document frequency
            # print(token)
            if token in dict_class:  # if token exists in dictionary
                dict_class[token] += 1  # then counter plus one
            else:
                dict_class.update({token: 1})  # if token didn't appear in dict., update count to one
    return dict_class


list_classDict = {}  # store all of classes' dictionary and their document freq.

for i in range(1,14):
    list_classDict[i] = getDictionary(i)


# ## Feature Selection: Likelihood ratio

# Feature Selection with Likelihood Ratio


def lrFeatureSelection(c, k):
    '''
    Parameter
    - c: target class number
    - k: output number of feature selection
    "Global dict./word" in this block means dict./word of all training data
    '''
    
    likelihood = {}  # store likelihood ratio of each word in global dictionary
    sortedLikelihood = {}  # store sorted likedlihood
    topWordEachClass = []  # store top k word for each class 

    for w in trainV:  # trainV is global dictionary

        # print(w)  # key
        # print(trainV[w])  # value: n_11+n_01

        if w in list_classDict[c]:  # count n_11 if global word exists in classX
            n_11 = list_classDict[c][w]
        else:
            n_11 = 0  # global word not found in classX

        n_01 = trainV[w] - n_11
        n_10 = 15 - n_11
        n_00 = 180 - n_01
        p_t = (n_11+n_01)/195
        p_1 = n_11/15
        p_2 = n_01/180

        result = -2*math.log( (p_t**n_11*(1-p_t)**n_10*p_t**n_01*(1-p_t)**n_00) / (p_1**n_11*(1-p_1)**n_10*p_2**n_01*(1-p_2)**n_00) )
        likelihood.update({w: result})

    sortedLikelihood = sorted(likelihood.items(), key=lambda x:x[1], reverse=True)[:k]  # sort likelihood in descending and get top k word
    topWordEachClass =[i[0] for i in sortedLikelihood]  # store top sorted word into list

    return topWordEachClass


# apply feature selection to all classes
'''
store top word of each class like:
{1:['a','b','c',...],
 2:['d','e','f',...],...}
'''
topWordClass = {}

for i in range(1,14):    
    topWordClass[i] = lrFeatureSelection(i, 30)


# store all classes' top words into flatten list with non-duplicated
trainV_featured = []

for c in topWordClass:
    for w in topWordClass[c]:
        # print(w)
        if w not in trainV_featured:
            trainV_featured.append(w)
trainV_featured

len(trainV_featured)  # 383 instead of 390


# ## Training with Naive Bayes


def concatAllDocInClass(D,c):  # store tokenized document for each class
    '''
    store document as:
    [['apple', 'banana'], ['cat', 'dog']]
    D: folder of training data
    c: id of class
    '''
    documents = []
    
    for doc in classesList[c]:
        with open("../pa2/"+ D + "/" + str(doc) + '.txt', 'rb') as txt:  # read txt files
            content = txt.read().decode('latin-1').replace('\n', '')  # decoding and ignore new line
            tokens = preprocess(content)
            documents.append(tokens)         
    
    return documents


def countTokensOfTerm(termListOfClass, dictOfAll):
    '''
    Count words of global dict. in each class
    （計算 global dict.的每個字在每個class出現幾次）
    - termListOfClass: 一個 class 合併後的所有 term，例如 concatAllDocInClass(D,c)
    - dictiOfAll: 所有 training data 萃取出來的字典
    '''    
    
    flattenList = [i for doc in termListOfClass for i in doc]  # flatten 2-d list to 1-d List
    dictOfClass = {x:flattenList.count(x) for x in flattenList}  # list to dictionary
    counter = {}
    for w in dictOfAll:
        if w in dictOfClass:
            counter.update({w: dictOfClass[w]})
        else:
            counter.update({w: 0})
    return counter


# training

countTrainData = 195  # total number of training documents
countDocsInClass = 15  # number of documents in each class

# store prior prob. for all classes
prior = {}  # they're all same(0.077)

# store every conditional probability of words in each class
condProb = {}  # 383x13

D = "data"

for c in range(1,14):
    # print(c)
    prior[c] = countDocsInClass/countTrainData
    text_c = concatAllDocInClass(D,c)
    lenText = sum(len(t) for t in text_c)  # total text length for each class(words may be repeated)
    term_count_each_class = countTokensOfTerm(text_c, trainV_featured)  # 算global dict.的字在這個class出現幾次
    # print(term_count_ineachClass)
    
    eachClassCondProb = {}
    
    for t in trainV_featured:
        # print(t)
        n_tct_smooth = term_count_each_class[t]+1  # 分子
        eachClassCondProb.update({t: (n_tct_smooth / (lenText+len(trainV_featured)))})
    
    condProb[c] = eachClassCondProb


# # Testing


def applyNaiveBayes(testingDoc):
    '''
    testingDoc: index of testing doc
    '''
    with open('../pa2/data/' + str(testingDoc) + '.txt', 'rb') as txt:
        content = txt.read().decode('latin-1').replace('\n', '')  # decoding and ignore new line
        tokens = preprocess(content)
    
    W = []  # testingDoc ⋂ training vocabulary
    for w in tokens:
        if w in trainV_featured:
            W.append(w)  # 取出認得的字 for 之後計算分數
    
    score = {}  # score of this testing document in each class
    for i in range(1, 14):
        score[i] = math.log(prior[i])  # basic
        for t in W:
            score[i] += math.log(condProb[i][t])  # add conditional probability

    return max(score, key=score.get)


# list of testing file id
test_id = [17,18,20,21,22,23,24,25,26,27,28,30,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,84,85,87,88,89,90,91,93,94,95,96,97,98,99,101,103,104,105,106,107,108,109,110,111,112,114,116,117,118,119,120,121,122,123,124,125,126,127,128,129,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,242,246,247,249,251,252,253,257,259,261,263,264,265,266,267,268,269,270,271,272,273,274,276,277,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,297,298,299,300,302,303,306,307,310,311,312,313,314,318,319,322,323,329,330,331,332,333,334,335,336,339,340,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,398,399,400,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,444,446,447,448,449,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,467,468,469,470,471,472,473,474,475,476,477,478,479,481,482,483,484,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,514,515,516,517,518,519,521,522,524,525,528,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,572,577,579,580,587,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,636,637,638,639,640,641,642,643,644,645,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,681,682,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,701,703,707,710,711,712,713,714,715,716,717,718,721,725,727,728,729,734,736,737,738,739,741,742,743,745,746,747,748,749,750,753,758,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,782,783,784,785,786,787,788,789,790,791,792,793,795,796,797,800,802,803,804,805,806,807,808,809,810,811,814,816,827,834,835,836,837,838,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,996,997,1000,1001,1002,1004,1008,1010,1017,1018,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095]

# apply Naive Bayes on testing file
apply = [applyNaiveBayes(i) for i in test_id]

# store result as dataframe
output = pd.DataFrame(apply, index=test_id, columns=['Value'])

# output result dataframe into csv
output.to_csv('hw3_final_r10725048.csv', index_label = ['Id'])
