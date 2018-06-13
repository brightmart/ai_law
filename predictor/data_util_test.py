# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import jieba
import re
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"

imprisonment_mean=26.2
imprisonment_std=33.5

def token_string_as_list(string,tokenize_style='word'):
    #string=string.decode("utf-8")
    string=replace_money_value(string)  #TODO add normalize number ADD 2018.06.11

    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt

def replace_money_value(string):
    moeny_list = [1,2,5,7,10, 20, 30,50, 100, 200, 500, 800,1000, 2000, 5000,7000, 10000, 20000, 50000, 80000,100000,200000, 500000, 1000000,3000000,5000000,1000000000]
    double_patten = r'\d+\.\d+'
    int_patten = r'[\u4e00-\u9fa5,，.。；;]\d+[元块万千百十余，,。.;；]'
    doubles=re.findall(double_patten,string)
    ints=re.findall(int_patten,string)
    ints=[a[1:-1] for a in ints]
    #print(doubles+ints)
    sub_value=0
    for value in (doubles+ints):
        for money in moeny_list:
            if money >= float(value):
                sub_value=money
                break
        string=re.sub(str(value),str(sub_value),string)
    return string

def pad_truncate_list(x_list, maxlen, value=0,truncating='pre',padding='pre'):
    """
    :param x_list:e.g. [1,10,3,5,...]
    :return:result_list:a new list,length is maxlen
    """
    result_list=[0 for i in range(maxlen)] #[0,0,..,0]
    length_input=len(x_list)
    if length_input>maxlen: #need to trancat===>no need to pad
        start_point = (length_input - maxlen)
        x_list=x_list[start_point:]
        for i, element in enumerate(x_list):
            result_list[i] = element
    else:#sequence is to short===>need to pad something===>no need to trancat. [1,2,3], max_len=1000.
        x_list.reverse() #[3,2,1]
        for i in range(length_input):
            result_list[i] = x_list[i]
        result_list.reverse()
    return result_list

#x_list=[1,2,3] #[0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
#x_list=[1,2,3,4,5,6,7,8,9,10,11,12,13]#===>[4,5,6,7,8,9,10,11,12,13]
#result_list=pad_truncate_list(x_list, 10, value=0,truncating='pre',padding='pre')
#print(result_list)



def load_word_vocab(file_path):
    """
    load vocab of word given file path
    :param file_path:
    :return: a dict, named:vocab_word2index
    """
    file_object=open(file_path,mode='r',encoding='utf8')
    lines=file_object.readlines()
    vocab_word2index={}
    vocab_word2index[_PAD] = PAD_ID
    vocab_word2index[_UNK] = UNK_ID
    for i,line in enumerate(lines):
        line=line.strip() #.decode("utf-8")
        if "::" in line:
            word=":"
        else:
            word,_=line.split(":") #wor,_="元:272339"
        vocab_word2index[word] = i + 2
    return vocab_word2index

def load_label_dict_accu(file_path):
    """
     load vocab of label given file path
     :param file_path:
     :return: a dict, named:label2index_dict
     """
    file_object = open(file_path, mode='r', encoding='utf8')
    lines = file_object.readlines()
    label2index_dict = {}
    for i, label in enumerate(lines):
        label = label.strip() #.decode("utf-8")
        label2index_dict[label] = i
    return label2index_dict

def load_label_dict_article(file_path):
    """
     load vocab of label given file path
     :param file_path:
     :return: a dict, named:label2index_dict
     """
    file_object = open(file_path, mode='r', encoding='utf8')
    lines = file_object.readlines()
    label2index_dict = {}
    for i, label in enumerate(lines):
        label = int(label.strip())
        label2index_dict[label] = i
    return label2index_dict