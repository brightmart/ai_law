# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import jieba
import random

def transform_format(source_file,target_file):
    """
    transform source format to target format
    :param source_file: source file
    :return:
    """
    #1.read data
    #2.loop for each line
    #3.process it and write to file system
    source_object=open(source_file,'r')
    target_object=open(target_file,'a')
    lines=source_object.readlines()
    random.shuffle(lines)
    for i,line in enumerate(lines):
        if i%10000==0:
            print(i)
        json_string=json.loads(line.strip())
        facts=json_string['fact']
        facts=token_string_as_list(facts)
        accusation_list=json_string['meta']['accusation']
        string_accusation=''
        for accusation in accusation_list:
            string_accusation+=' __label__'+accusation
        target_object.write(" ".join(facts)+string_accusation+"\n")

    target_object.close()
    source_object.close()



def token_string_as_list(string,tokenize_style='word'):
    string=string.decode("utf-8")
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt

source_file='data/data_train.json'
target_file='data/data_train_accusation.txt'
transform_format(source_file,target_file)