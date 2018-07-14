# -*- coding: utf-8 -*-
import codecs
import random
import json

def search(keyword,traning_data_path):
    """
    given keyword,get facts.
    :param keyword: a string
    :return:
    """
    #1.load data as lines
    train_file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    lines = train_file_object.readlines()
    random.shuffle(lines)
    #2.loop for each line, if match condition put to list.
    result_list=[]
    for i,line in enumerate(lines):
        if len(line)<10:continue
        json_string = json.loads(line.strip())
        fact = json_string['fact']
        accusation_list = json_string['meta']['accusation']
        article_list = json_string['meta']['relevant_articles']

        if keyword in accusation_list:
            result_list.append((fact,accusation_list,article_list))
    return result_list

keyword=u'爆炸'
fact_lists=search(keyword,'../data_big/cail2018_big_downsmapled.json')
for tuplee in fact_lists:
    fact,accusation_list,article_list=tuplee
    print(fact)
    print("----")
    for accusation in accusation_list:
        print(accusation)
    print("article_list:",article_list)
    print("-------------------------")
