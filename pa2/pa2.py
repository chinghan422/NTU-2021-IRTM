#!/usr/bin/env python
# coding: utf-8

import re
import os
import nltk
import math
import numpy as np
from numpy import dot
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')


def preprocess(text):  # 將pa1包成 preprocess function

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


def count(term, dictionary):  # count document frequency
    if term in dictionary:
        dictionary[term] += 1
    else:
        dictionary.update({term: 1})


'''
Term Frequency: 單篇文章出現頻率(document-wise)
For example,
Doc1: I am so so happy
-> tf(so, Doc1) = 2/5 = 0.4
'''


def tf(term, tokens):
    freq_dict = dict([(token, tokens.count(token)) for token in tokens])
    freq_query = freq_dict.get(term, 0)  # return default value '0' when key missing
    len_doc = sum(freq_dict.values())
    result = freq_query / len_doc
    return term, result


'''
Document Frequency: 出現在幾篇文章(collection-wise)
idf(Inverse Document Frequency): The idf of a rare term is high, and low for a frequent term
'''


def idf(term, documents):
    doc_count = 0
    for document in documents:
        if term in document:
            doc_count += 1
        result = math.log10(len(documents) / (doc_count + 1))
    return result


# 1. Construct a dictionary based on the terms extracted from the given documents

path = "./data"  # collection folder
files = os.listdir(path)  # get all files under collection folder

# store document as
# [['apple', 'banana'], ['cat', 'dog']]
documents = []

# {term: doc_fre} e.g. {'apple':2, 'banana':1}
dictionary = {}

# {term: index} e.g. {'apple': 41...}
key2Index = {}

# record txt file read sequence
read_sequence = []

for file in files:  # 遍歷資料夾
    if file.endswith(".txt"):
        read_sequence.append(file.split(".")[0])
        with open(path+"/"+file, 'rb') as txt:  # read txt files
            content = txt.read().decode('latin-1').replace('\n', '')  # decoding and ignore new line
            tokens = preprocess(content)
            documents.append(tokens)
                            
# print(read_sequence)
  
for document in documents:
    for token in document:  # counting doc. frequency
        count(token, dictionary)  # 計算 token 個數並記錄到 dictionary
        
# 寫入 dictionary.txt
print("Create dictionary")
dict_path = "./dictionary.txt"  # create dictionary.txt
f = open(dict_path, 'w')
f.write("t_index\tterm\tdf\r")  # header
# using enumerate to get index and items
for idx, key in enumerate(sorted(dictionary)):
    toWrite = "%d\t%s\t%s\r" % (idx+1, key, dictionary[key])
    key2Index.update({key: idx+1})
    f.write(toWrite)
f.close()

# print(key2Index)
# print(key2Index.get("determin"))


# 2. Transfer each document into a tf-idf unit vector

print("Calculate tf-idf")
output_path = "./output/"

for idx, document in enumerate(documents):
    print(str(idx) + ": " + read_sequence[idx])  # logging
    
    # output format
    output_dict = {}  # {idx: tf-idf}
    
    # unit vector
    unit_vec = 0
    
    f = open(output_path+'doc'+read_sequence[idx]+'.txt', 'w')
    
    for token in document:
        word, term_tf = tf(token, document)
        term_idf = idf(token, documents)
        tf_idf = term_tf*term_idf  # calculate tf-idf
        
        unit_vec += tf_idf**2

        print('%d\r%s\t%f\t%f\t%f' % (idx, word, term_tf, term_idf, tf_idf))
        
        # avoid output duplicate term in DocID.txt
        if key2Index.get(word) not in output_dict:
            output_dict.update({key2Index.get(word): tf_idf})
    
    unit_vec_root = math.sqrt(unit_vec)
    
    f.write(str(len(output_dict))+'\r')  # term amount
    f.write("t_index\ttf-idf\r")  # header
    
    for key in sorted(output_dict):  # sorted output dictionary
        f.write("%s\t%f\r" % (key, output_dict.get(key)/unit_vec_root))
    f.close()


# 3. Write a function cosine(Docx, Docy) which loads the tf-idf vectors of documents x & y and returns their cosine similarity

'''
Input two document id to calculate their cosine similarity
through tf-idf unit vector
'''

def cosine(Docx, Docy):
    dict_x = {}  # dictionary of document x
    dict_y = {}  # dictionary of document y

    with open('./output/doc' + str(Docx) + '.txt',
              'r') as txt:  # read file write to dictionary
        content_x = [
            i.replace("\n", "").split("\t") for i in txt.readlines()[2:]
        ]

    for item in content_x:
        dict_x.update({item[0]: item[1]})

    with open('./output/doc' + str(Docy) + '.txt',
              'r') as txt:  # read file write to dictionary
        content_y = [
            i.replace("\n", "").split("\t") for i in txt.readlines()[2:]
        ]

    for item in content_y:
        dict_y.update({item[0]: item[1]})

    for item in dict_x:  # check items in another dict(y), if non-exist->give 0
        if item not in dict_y:
            dict_y.update({item: 0})

    for item in dict_y:  # check item in another dict(x), if non-exist->give 0
        if item not in dict_x:
            dict_x.update({item: 0})

    x = []  # create list x[] to store tf-idf score of document x
    y = []  # create list y[] to store tf-idf score of document y

    for key in sorted(dict_x):  # call sorted dictionary x
        x.append(float(dict_x.get(key)))  # get value: string to float

    for key in sorted(dict_y):  # call sorted dictionary y
        y.append(float(dict_y.get(key)))  # get value of y
    
    x = np.array(x)  # convert list x to array for calculating cosine similarity
    y = np.array(y)  # convert list y to array for calculating cosine similarity
    
    similarity_scores = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))

    return similarity_scores


cosine(1, 1)

cosine(1, 2)
