# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle
import json
import jieba

PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"

imprisonment_mean=26.2
imprisonment_std=33.5
def load_data_multilabel(traning_data_path,valid_data_path,test_data_path,vocab_word2index, accusation_label2index,article_label2index,
                         deathpenalty_label2index,lifeimprisonment_label2index,sentence_len,name_scope='cnn',test_mode=False):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    # 1. use cache file if exist
    cache_data_dir = 'cache' + "_" + name_scope;cache_file =cache_data_dir+"/"+'train_valid_test.pik'
    print("cache_path:",cache_file,"train_valid_test_file_exists:",os.path.exists(cache_file))
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file from file system and return")
            return pickle.load(data_f)
    # 2. read source file
    train_file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    valid_file_object = codecs.open(valid_data_path, mode='r', encoding='utf-8')
    test_data_obejct = codecs.open(test_data_path, mode='r', encoding='utf-8')
    train_lines = train_file_object.readlines()
    valid_lines=valid_file_object.readlines()
    test_lines=test_data_obejct.readlines()
    random.shuffle(train_lines)

    if test_mode:
        train_lines=train_lines[0:1000]
    # 3. transform to train/valid data to standardized format
    train=transform_data_to_index(train_lines, vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len)
    valid=transform_data_to_index(valid_lines, vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len)
    test=transform_data_to_index(test_lines, vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len)

    # 4. save to file system if vocabulary of words not exists
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to dump train/valid/test data to file sytem.")
            pickle.dump((train,valid,test),data_f)
    return train,valid,test


def transform_data_to_index(lines,vocab_word2index,accusation_label2index,article_label2index,deathpenalty_label2index,lifeimprisonment_label2index,sentence_len,reverse_flag=False):
    """
    transform data to index using vocab and label dict.
    :param lines:
    :param vocab_word2index:
    :param accusation_label2index:
    :param article_label2index:
    :param deathpenalty_label2index:
    :param lifeimprisonment_label2index:
    :param sentence_len: max sentence length
    :return:
    """
    X = []
    Y_accusation = []  # discrete
    Y_article = []  # discrete
    Y_deathpenalty = []  # discrete
    Y_lifeimprisonment = []  # discrete
    Y_imprisonment = []  # continuous
    #for k,v in accusation_label2index.items():
    #    print(k);print(v)
    accusation_label_size=len(accusation_label2index)
    article_lable_size=len(article_label2index)
    for i, line in enumerate(lines):
        if i%10000==0:
            print("i:", i)
        json_string = json.loads(line.strip())

        # 1. transform input x.discrete
        facts = json_string['fact']
        input_list = token_string_as_list(facts)  # tokenize
        x = [vocab_word2index.get(x, UNK_ID) for x in input_list]  # transform input to index
        X.append(x)

        # 2. transform accusation.discrete
        accusation_list = json_string['meta']['accusation']
        accusation_list = [accusation_label2index[label] for label in accusation_list]
        y_accusation = transform_multilabel_as_multihot(accusation_list, accusation_label_size)
        Y_accusation.append(y_accusation)

        #print("article_label2index:")
        #for k,v in article_label2index.items():
        #    print(k,v)
        # 3.transform relevant article.discrete
        article_list = json_string['meta']['relevant_articles']
        #print("article_list[0]:",article_list[0],type(article_list[0]))
        article_list = [article_label2index[label] for label in article_list]
        y_article = transform_multilabel_as_multihot(article_list, article_lable_size)
        Y_article.append(y_article)

        # 4.transform death penalty.discrete
        death_penalty = json_string['meta']['term_of_imprisonment']['death_penalty']  # death_penalty
        death_penalty = deathpenalty_label2index[death_penalty]
        y_deathpenalty = transform_multilabel_as_multihot(death_penalty, 2)
        Y_deathpenalty.append(y_deathpenalty)

        # 5.transform life imprisonment.discrete
        life_imprisonment = json_string['meta']['term_of_imprisonment']['life_imprisonment']
        life_imprisonment = lifeimprisonment_label2index[life_imprisonment]
        y_lifeimprisonment = transform_multilabel_as_multihot(life_imprisonment, 2)
        Y_lifeimprisonment.append(y_lifeimprisonment)

        # 6.transform imprisonment.continuous
        imprisonment = json_string['meta']['term_of_imprisonment']['imprisonment']  # continuous value like:10
        y_imprisonment = float((imprisonment-imprisonment_mean)/imprisonment_std)
        Y_imprisonment.append(y_imprisonment)

    X = pad_sequences(X, maxlen=sentence_len, value=0.,truncating='pre')  # padding to max length.remove sequence that longer than max length from beginning.
    #reverse
    if reverse_flag:
        X_=np.zeros(X.shape)
        for i, element in enumerate(X):
            e = list(element);
            e.reverse()
            X_[i] = np.array(e)
        X=X_

    data = (X, Y_accusation, Y_article, Y_deathpenalty, Y_lifeimprisonment, Y_imprisonment)
    return data

def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(data_path,training_data_path,vocab_size,name_scope='cnn',test_mode=False):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    #0.if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            print("going to load cache file.vocab of words and labels")
            return pickle.load(data_f)
    else:
        vocab_word2index={}
        vocab_word2index[_PAD]=PAD_ID
        vocab_word2index[_UNK]=UNK_ID

        accusation_label2index={}
        articles_label2index={}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines=file_object.readlines()
        random.shuffle(lines)
        if test_mode:
           lines=lines[0:10000]
        #2.loop each line,put to counter
        c_inputs=Counter()
        c_accusation_labels=Counter()
        c_article_labels=Counter()
        for i,line in enumerate(lines):
            if i%10000==0:
                print(i)
            json_string = json.loads(line.strip())
            facts = json_string['fact']
            input_list = token_string_as_list(facts)
            c_inputs.update(input_list)

            accusation_list = json_string['meta']['accusation']
            c_accusation_labels.update(accusation_list)

            article_list = json_string['meta']['relevant_articles']
            c_article_labels.update(article_list)

        #3.get most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        word_freq_file=codecs.open(cache_vocabulary_label_pik+"/"+'word_freq.txt',mode='a',encoding='utf-8')
        #put those words to dict
        for i,tuplee in enumerate(vocab_list):
            word,freq=tuplee
            word_freq_file.write(word+":"+str(freq)+"\n")
            vocab_word2index[word]=i+2

        #4.1 accusation and its frequency.
        accusation_freq_file=codecs.open(cache_vocabulary_label_pik+"/"+'accusation_freq.txt',mode='a',encoding='utf-8')
        accusation_label_list=c_accusation_labels.most_common()
        for i,tuplee in enumerate(accusation_label_list):
            label,freq=tuplee
            accusation_freq_file.write(label+":"+str(freq)+"\n")

        #4.2 accusation dict
        accusation_voc_file=data_path+"/accu.txt"
        accusation_voc_object=codecs.open(accusation_voc_file,mode='r',encoding='utf-8')
        accusation_voc_lines=accusation_voc_object.readlines()
        for i,accusation_name in enumerate(accusation_voc_lines):
            accusation_name=accusation_name.strip()
            accusation_label2index[accusation_name]=i

        #5.1 relevant article(law) and its frequency
        article_freq_file=codecs.open(cache_vocabulary_label_pik+"/"+'article_freq.txt',mode='a',encoding='utf-8')
        article_label_list=c_article_labels.most_common()
        for j,tuplee in enumerate(article_label_list):
            label,freq=tuplee
            article_freq_file.write(str(label)+":"+str(freq)+"\n")

        #5.2 relevant article dict
        article_voc_file=data_path+"/law.txt"
        article_voc_object=codecs.open(article_voc_file,mode='r',encoding='utf-8')
        article_voc_lines=article_voc_object.readlines()
        for i,law_id in enumerate(article_voc_lines):
            law_id=int(law_id.strip())
            articles_label2index[law_id]=i

        #6.save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                print("going to save cache file of vocab of words and labels")
                pickle.dump((vocab_word2index, accusation_label2index,articles_label2index), data_f)

    #7.close resources
    word_freq_file.close()
    accusation_freq_file.close()
    article_freq_file.close()
    print("create_vocabulary.ended")
    return vocab_word2index, accusation_label2index,articles_label2index

def token_string_as_list(string,tokenize_style='word'):
    #string=string.decode("utf-8")
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt

def get_part_validation_data(valid,num_valid=2000):
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment=valid
    number_examples=len(valid_X)
    permutation = np.random.permutation(number_examples)[0:num_valid]
    valid_X2, valid_Y_accusation2, valid_Y_article2, valid_Y_deathpenalty2, valid_Y_lifeimprisonment2, valid_Y_imprisonment2=[],[],[],[],[],[]
    for index in permutation :
        valid_X2.append(valid_X[index])
        valid_Y_accusation2.append(valid_Y_accusation[index])
        valid_Y_article2.append(valid_Y_article[index])
        valid_Y_deathpenalty2.append(valid_Y_deathpenalty[index])
        valid_Y_lifeimprisonment2.append(valid_Y_lifeimprisonment[index])
        valid_Y_imprisonment2.append(valid_Y_imprisonment[index])
    return valid_X2,valid_Y_accusation2,valid_Y_article2,valid_Y_deathpenalty2,valid_Y_lifeimprisonment2,valid_Y_imprisonment2

#training_data_path='../data/sample_multiple_label3.txt'
#vocab_size=100
#create_voabulary(training_data_path,vocab_size)
