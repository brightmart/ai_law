# -*- coding: utf-8 -*-
import random
import json
import codecs

def construct_validation_set(valid_file_path,training_file_path,target_valid_file_path):
    # 1. read training/validation, get a id for each line from training
    valid_file_object = codecs.open(valid_file_path, mode='r', encoding='utf-8')
    valid_lines=valid_file_object.readlines()
    training_object = codecs.open(training_file_path, mode='r', encoding='utf-8')
    train_lines=training_object.readlines()
    target_valid_file_path = codecs.open(target_valid_file_path, mode='a', encoding='utf-8')

    # 2. loop each validation, remove those line exist in training, save to file system.
    dict_meta_id={}
    # training
    for j,linee in enumerate(train_lines):
        json_string = json.loads(linee.strip())
        accusation_list = json_string['meta']['accusation']
        article_list = json_string['meta']['relevant_articles']
        fact = json_string['fact']
        unique_id=str(hash(str(accusation_list)))+"_"+str(hash(str(article_list)))+"_"+str(hash(fact[0:20]))
        dict_meta_id[unique_id]=unique_id

    for i,line in enumerate(valid_lines):
        json_stringg = json.loads(line.strip())
        accusation_listt = json_stringg['meta']['accusation']
        article_listt = json_stringg['meta']['relevant_articles']
        factt = json_stringg['fact']
        unique_idd=str(hash(str(accusation_listt)))+"_"+str(hash(str(article_listt)))+"_"+str(hash(factt[0:20]))
        if unique_idd not in dict_meta_id: #save
            target_valid_file_path.write(line)
        else: # print info for debug
            print(line)

    target_valid_file_path.close()
    training_object.close()
    valid_file_object.close()

valid_file_path='../data/data_valid.json'
training_file_path='../data_big/cail2018_big.json' #cail2018_big_downsmapled.json
target_valid_file_path='../data/data_valid_checked.json'
construct_validation_set(valid_file_path,training_file_path,target_valid_file_path)