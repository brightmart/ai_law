# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import csv
#from data_util import *
import json
import jieba
import random
def length_of_sentence(file_path):
    #1.read data
    file_object = open(file_path, 'r')
    lines=file_object.readlines()
    total_length=0
    count=0
    max_length=0
    mini_length=10000
    length_dict={50:0,75:0,100:0,150:0,200:0,300:0,400:0,500:0,600:0,700:0}
    random.shuffle(lines)
    lines=lines[0:10000]
    for i, line in enumerate(lines):
        json_string=json.loads(line)
        facts=json_string['fact']
        x_list=token_string_as_list(facts,tokenize_style='word')
        length=len(x_list)
        total_length=total_length+length
        #print("length:",length)
        if length>=max_length:
            max_length=length
        if length<=mini_length:
            mini_length=length
        count=count+1
        if i%10000==0:
            print(i)
        if length<50:
            length_dict[50]=length_dict[50]+1
        elif length<75:
            length_dict[75] = length_dict[75] + 1
        elif length<100:
            length_dict[100] = length_dict[100] + 1
        elif length<150:
            length_dict[150] = length_dict[150] + 1
        elif length<200:
            length_dict[200] = length_dict[200] + 1
        elif length<300:
            length_dict[300] = length_dict[300] + 1
        elif length<400:
            length_dict[400] = length_dict[400] + 1
        elif length<500:
            length_dict[500] = length_dict[500] + 1
        elif length<600:
            length_dict[600] = length_dict[600] + 1
        else:
            length_dict[700] = length_dict[700] + 1
    print("length_dict1:",length_dict)
    length_dict={k:float(v)/float(count) for k,v in length_dict.items()}
    print("length_dict2:", length_dict)
    file_object.close()

    avg_length=(float(total_length))/float(count)
    print("avg length:",avg_length)
    print("mini_length:",mini_length)
    print("max_length:",max_length)

    print sorted(length_dict.items(), key=lambda d: d[0])


def label_length_of_sentence(file_path):
    #1.read data
    file_object = open(file_path, 'r')
    lines=file_object.readlines()
    total_length=0
    count=0
    max_length=0
    mini_length=10000
    length_dict={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    random.shuffle(lines)
    lines=lines[0:10000]
    for i, line in enumerate(lines):
        json_string=json.loads(line)
        relevant_articles=json_string['meta']['accusation']
        length=len(relevant_articles)
        total_length=total_length+length
        #print("length:",length)
        if length>=max_length:
            max_length=length
        if length<=mini_length:
            mini_length=length
        count=count+1
        if i%10000==0:
            print(i)
        if length<=1:
            length_dict[1]=length_dict[1]+1
        elif length<2:
            length_dict[2] = length_dict[2] + 1
        elif length<3:
            length_dict[3] = length_dict[3] + 1
        elif length<4:
            length_dict[4] = length_dict[4] + 1
        elif length<5:
            length_dict[5] = length_dict[5] + 1
        elif length<6:
            length_dict[6] = length_dict[6] + 1
        else:
            length_dict[7] = length_dict[7] + 1
    print("length_dict1:",length_dict)
    length_dict={k:float(v)/float(count) for k,v in length_dict.items()}
    print("length_dict2:", length_dict)
    file_object.close()

    avg_length=(float(total_length))/float(count)
    print("avg length:",avg_length)
    print("mini_length:",mini_length)
    print("max_length:",max_length)

    print sorted(length_dict.items(), key=lambda d: d[0])

def token_string_as_list(string,tokenize_style='word'):
    string=string.decode("utf-8")
    string=string.replace("***","*")
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string) #cut_all=True

    listt=[x for x in listt if x.strip()]
    return listt

file_path='../data/data_train.json'
label_length_of_sentence(file_path)

