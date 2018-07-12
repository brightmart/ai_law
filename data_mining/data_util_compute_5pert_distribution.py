# -*- coding: utf-8 -*-
import codecs
import random
from collections import Counter
import json

def xxx(training_data_path,name_scope):
   cache_vocabulary_label_pik = '../cache' + "_" + name_scope  # path to save cache
   #1.read lines
   file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
   lines = file_object.readlines()
   #length=len(lines)
   random.shuffle(lines)
   #five_percentage=int(length*0.10)
   #print("five_percentage:",five_percentage)
   #lines=lines[0:five_percentage]

   #2.compute frequency
   c_accusation_labels = Counter()
   c_article_labels = Counter()
   for i, line in enumerate(lines):
      if i % 10000 == 0:
         print(i)
      json_string = json.loads(line.strip())
      accusation_list = json_string['meta']['accusation']
      c_accusation_labels.update(accusation_list)
      article_list = json_string['meta']['relevant_articles']
      c_article_labels.update(article_list)

   # 3.write accusation and its frequency.
   accusation_freq_file = codecs.open(cache_vocabulary_label_pik + "/" + 'accusation_freq_valid.txt', mode='a',encoding='utf-8')
   accusation_label_list = c_accusation_labels.most_common()
   for i, tuplee in enumerate(accusation_label_list):
      label, freq = tuplee
      accusation_freq_file.write(label + ":" + str(freq) + "\n")
   accusation_freq_file.close()

   # 4. write relevant article(law) and its frequency
   article_freq_file = codecs.open(cache_vocabulary_label_pik + "/" + 'article_freq_valid.txt', mode='a', encoding='utf-8')
   article_label_list = c_article_labels.most_common()
   for j, tuplee in enumerate(article_label_list):
      label, freq = tuplee
      article_freq_file.write(str(label) + ":" + str(freq) + "\n")
   article_freq_file.close()


training_data_path='../data/data_valid.json' #''../data_big/cail2018_big.json'
name_scope='text_cnn'
xxx(training_data_path,name_scope)


