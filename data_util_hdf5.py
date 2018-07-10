# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
import multiprocessing

from collections import Counter
import os
#import pickle
import cPickle as pickle
import h5py

import json
import jieba

PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"

imprisonment_mean=26.2
imprisonment_std=33.5
from predictor.data_util_test import  pad_truncate_list

def build_chunk(lines, chunk_num=10):
    """
    split list into sub lists:分块
    :param lines: total thing
    :param chunk_num: num of chunks
    :return: return chunks but the last chunk may not be equal to chunk_size
    """
    total = len(lines)
    chunk_size = float(total) / float(chunk_num + 1)
    chunks = []
    for i in range(chunk_num + 1):
        if i == chunk_num:
            chunks.append(lines[int(i * chunk_size):])
        else:
            chunks.append(lines[int(i * chunk_size):int((i + 1) * chunk_size)])
    return chunks

def load_data_multilabel(traning_data_path,valid_data_path,test_data_path,vocab_word2index, accusation_label2index,article_label2index,
                         deathpenalty_label2index,lifeimprisonment_label2index,sentence_len,name_scope='cnn',test_mode=False,valid_number=120000,#12000
                         test_number=10000,process_num=30,tokenize_style='word'):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    # 1. use cache file if exist
    cache_data_dir = 'cache' + "_" + name_scope;cache_file =cache_data_dir+"/"+'train_valid_test.h5'
    print("cache_path:",cache_file,"train_valid_test_file_exists:",os.path.exists(cache_file))
    if os.path.exists(cache_file):
        #with open(cache_file, 'rb') as data_f:
        print("going to load cache file from file system and return")
        f = h5py.File(cache_file, 'r')
        #train=f['train']
        #valid=f['valid']
        #test=f['test']
        X_array=f['train_X_array']
        Y_accusation=f['train_Y_accusation']
        Y_article=f['train_Y_article']
        Y_deathpenalty=f['train_Y_deathpenalty']
        Y_lifeimprisonment=f['train_Y_lifeimprisonment']
        Y_imprisonment=f['train_Y_imprisonment']
        weights_accusation=f['train_weights_accusation']
        weights_article=f['train_weights_article']
        train=np.array(X_array),np.array(Y_accusation),np.array(Y_article),np.array(Y_deathpenalty),np.array(Y_lifeimprisonment),np.array(Y_imprisonment),np.array(weights_accusation),np.array(weights_article)

        valid_X_array=f['valid_X_array']
        valid_Y_accusation=f['valid_Y_accusation']
        valid_Y_article=f['valid_Y_article']
        valid_Y_deathpenalty=f['valid_Y_deathpenalty']
        valid_Y_lifeimprisonment=f['valid_Y_lifeimprisonment']
        valid_Y_imprisonment=f['valid_Y_imprisonment']
        valid_weights_accusation=f['valid_weights_accusation']
        valid_weights_article=f['valid_weights_article']
        valid=np.array(valid_X_array),np.array(valid_Y_accusation),np.array(valid_Y_article),np.array(valid_Y_deathpenalty),np.array(valid_Y_lifeimprisonment),np.array(valid_Y_imprisonment),np.array(valid_weights_accusation),np.array(valid_weights_article)

        test_X_array=f['test_X_array']
        test_Y_accusation=f['test_Y_accusation']
        test_Y_article=f['test_Y_article']
        test_Y_deathpenalty=f['test_Y_deathpenalty']
        test_Y_lifeimprisonment=f['test_Y_lifeimprisonment']
        test_Y_imprisonment=f['test_Y_imprisonment']
        test_weights_accusation=f['test_weights_accusation']
        test_weights_article=f['test_weights_article']
        test=np.array(test_X_array),np.array(test_Y_accusation),np.array(test_Y_article),np.array(test_Y_deathpenalty),np.array(test_Y_lifeimprisonment),np.array(test_Y_imprisonment),np.array(test_weights_accusation),np.array(test_weights_article)

        f.close()
        return train,valid,test
        #return pickle.load(data_f)
    # 2. read source file
    train_file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    train_lines_original = train_file_object.readlines()
    random.shuffle(train_lines_original)

    if test_mode:
        train_lines_original = train_lines_original[0:1000 * 100]  # 1000
        valid_number=20000

    number_examples=len(train_lines_original)
    valid_start=number_examples-(valid_number+test_number)

    train_lines=train_lines_original[0:valid_start]
    valid_lines=train_lines_original[valid_start:valid_start+valid_number]#valid_lines=valid_file_object.readlines()
    test_lines=train_lines_original[valid_start+valid_number:]#test_lines=test_data_obejct.readlines()

    print("length of train_lines:",len(train_lines),";length of valid_lines:",len(valid_lines),";length of test_lines:",len(test_lines))

    # 3. transform to train/valid data to standardized format #TODO change to multi-processing version for train
    ##############below is for multi-processing########################################################################################################
    # 3.1 get chunks as list.
    chunks = build_chunk(train_lines, chunk_num=process_num - 1)
    pool = multiprocessing.Pool(processes=process_num)
    # 3.2 use multiprocessing to handle different chunk. each chunk will be transformed and save to file system.
    for chunk_id, each_chunk in enumerate(chunks):
        file_namee=cache_data_dir+'/' + "training_data_temp_" + str(chunk_id)+".pik" #cache_data_dir+'/' + "training_data_temp_" + str(chunk_id)+".pik"
        print("start multi-processing:",chunk_id,file_namee)
        # apply_async
        pool.apply_async(transform_data_to_index,args=(each_chunk, file_namee, vocab_word2index, accusation_label2index,article_label2index,deathpenalty_label2index, lifeimprisonment_label2index,sentence_len,'train',name_scope,tokenize_style))  # a common function named 'task' will be invoked for each file; args include sub list and name of target file.
    pool.close()
    pool.join()
    print("finish reduce stage...")

    # 3.3 merge sub file to final file.
    X, Y_accusation, Y_article, Y_deathpenalty, Y_lifeimprisonment, Y_imprisonment, weights_accusation, weights_article=[],[],[],[],[],[],[],[]
    for chunk_id in range(process_num):
        file_name =cache_data_dir+'/' + "training_data_temp_" + str(chunk_id)+".pik"
        with open(file_name, 'rb') as data_f:#rb
            X_, Y_accusation_, Y_article_, Y_deathpenalty_, Y_lifeimprisonment_, Y_imprisonment_, weights_accusation_, weights_article_=pickle.load(data_f)
            X.extend(X_);Y_accusation.extend(Y_accusation_);Y_article.extend(Y_article_);Y_deathpenalty.extend(Y_deathpenalty_);Y_lifeimprisonment.extend(Y_lifeimprisonment_)
            Y_imprisonment.extend(Y_imprisonment_);weights_accusation.extend(weights_accusation_);weights_article.extend(weights_article_)
            command = 'rm ' + file_name
            os.system(command)
    ##############above is for multi-processing##########################################################################################################

    #train=transform_data_to_ind
    # ex(train_lines, vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len,'train',name_scope)
    valid=transform_data_to_index(valid_lines, None,vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len,'valid',name_scope,tokenize_style)
    valid_X_array, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment, valid_weights_accusation, valid_weights_article=valid

    test=transform_data_to_index(test_lines, None,vocab_word2index, accusation_label2index, article_label2index,deathpenalty_label2index, lifeimprisonment_label2index, sentence_len,'test',name_scope,tokenize_style)
    test_X_array, test_Y_accusation, test_Y_article, test_Y_deathpenalty, test_Y_lifeimprisonment, test_Y_imprisonment, test_weights_accusation, test_weights_article=test

    X_array=np.array(X)
    train= X_array, Y_accusation, Y_article, Y_deathpenalty, Y_lifeimprisonment, Y_imprisonment, weights_accusation, weights_article

    # 4. save to file system if vocabulary of words not exists
    if 1==2:#if not os.path.exists(cache_file): #TODO TODO TODO test 2018.07.05
        #with open(cache_file, 'ab') as data_f:
        print("going to dump train/valid/test data to file sytem!")
            #pickle.dump((train,valid,test),data_f,protocol=pickle.HIGHEST_PROTOCOL) #TEMP REMOVED. ,protocol=2
        f = h5py.File(cache_file, 'w')
        f['train_X_array']=X_array
        f['train_Y_accusation']=Y_accusation
        f['train_Y_article']=Y_article
        f['train_Y_deathpenalty']=Y_deathpenalty
        f['train_Y_lifeimprisonment']=Y_lifeimprisonment
        f['train_Y_imprisonment']=Y_imprisonment
        f['train_weights_accusation']=weights_accusation
        f['train_weights_article']=weights_article

        f['valid_X_array']=valid_X_array
        f['valid_Y_accusation']=valid_Y_accusation
        f['valid_Y_article']=valid_Y_article
        f['valid_Y_deathpenalty']=valid_Y_deathpenalty
        f['valid_Y_lifeimprisonment']=valid_Y_lifeimprisonment
        f['valid_Y_imprisonment']=valid_Y_imprisonment
        f['valid_weights_accusation']=valid_weights_accusation
        f['valid_weights_article']=valid_weights_article

        f['test_X_array']=test_X_array
        f['test_Y_accusation']=test_Y_accusation
        f['test_Y_article']=test_Y_article
        f['test_Y_deathpenalty']=test_Y_deathpenalty
        f['test_Y_lifeimprisonment']=test_Y_lifeimprisonment
        f['test_Y_imprisonment']=test_Y_imprisonment
        f['test_weights_accusation']=test_weights_accusation
        f['test_weights_article']=test_weights_article
        f.close()

    return train ,valid,test

splitter=':'
num_mini_examples=3500 #1900


def transform_data_to_index(lines,target_file_path,vocab_word2index,accusation_label2index,article_label2index,deathpenalty_label2index,lifeimprisonment_label2index,
                            sentence_len,data_type,name_scope,tokenize_style):#reverse_flag=False
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
    print(data_type,"transform_data_to_index.####################.start.")
    X = []
    Y_accusation = []  # discrete
    Y_article = []  # discrete
    Y_deathpenalty = []  # discrete
    Y_lifeimprisonment = []  # discrete
    Y_imprisonment = []  # continuous
    weights_accusation=[]
    weights_article=[]
    accusation_label_size=len(accusation_label2index)
    article_lable_size=len(article_label2index)

    # load frequency of accu and relevant articles, so that we can copy those data with label are few. ADD 2018-05-29
    accusation_freq_dict, article_freq_dict = load_accusation_articles_freq_dict(accusation_label2index,article_label2index, name_scope)

    for i, line in enumerate(lines):
        if i%10000==0:print(data_type,"i:", i)
        json_string = json.loads(line.strip())

        # 1. transform input x.discrete
        facts = json_string['fact']
        input_list = token_string_as_list(facts,tokenize_style=tokenize_style)  # tokenize
        x = [vocab_word2index.get(x, UNK_ID) for x in input_list]  # transform input to index
        if i % 100000 == 0:print(i,"#######transform_data_to_index.x:",x)
        x=pad_truncate_list(x, sentence_len) #ADD 2018.05.24

        # 2. transform accusation.discrete
        accusation_list = json_string['meta']['accusation']
        if i % 50000 == 0:print(i,"#######transform_data_to_index.accusation_list(string):",accusation_list)
        accusation_list = [accusation_label2index[label] for label in accusation_list] #TODO add 2018.07.04
        if i % 50000 == 0:print(i,"#######transform_data_to_index.accusation_list(index):",accusation_list)

        #print(i,"accusation_list",accusation_list)
        y_accusation = transform_multilabel_as_multihot(accusation_list, accusation_label_size)

        # 3.transform relevant article.discrete
        article_list = json_string['meta']['relevant_articles']
        if i % 50000 == 0:print(i,"#######transform_data_to_index.article_list(string):",article_list)
        article_list = [article_label2index[int(label)] for label in article_list] #label-->int(label) #2018-06-13
        if i % 50000 == 0:print(i,"#######transform_data_to_index.article_list(index):",article_list)

        y_article = transform_multilabel_as_multihot(article_list, article_lable_size)

        # 4.transform death penalty.discrete
        death_penalty = json_string['meta']['term_of_imprisonment']['death_penalty']  # death_penalty
        death_penalty = deathpenalty_label2index[death_penalty]
        y_deathpenalty = transform_multilabel_as_multihot(death_penalty, 2)

        # 5.transform life imprisonment.discrete
        life_imprisonment = json_string['meta']['term_of_imprisonment']['life_imprisonment']
        life_imprisonment = lifeimprisonment_label2index[life_imprisonment]
        y_lifeimprisonment = transform_multilabel_as_multihot(life_imprisonment, 2)

        # 6.transform imprisonment.continuous
        imprisonment = json_string['meta']['term_of_imprisonment']['imprisonment']  # continuous value like:10

        # OVER-SAMPLING:if it is training data, copy labels that are few based on their frequencies.
        num_copy = 1
        weight_accusation=1.0
        weight_artilce=1.0
        if data_type == 'train': #set specially weight and copy some examples when it is training data.

            freq_accusation =min([accusation_freq_dict[xx] for xx in accusation_list]) # accusation_freq_dict[accusation_list[0]]
            freq_article =min([article_freq_dict[xxx] for xxx in article_list]) # article_freq_dict[article_list[0]]

            if freq_accusation <= num_mini_examples or freq_article <= num_mini_examples:
                freq=(freq_accusation+freq_article)/2
                num_copy=int(max(3,num_mini_examples/freq))
                if i%1000==0: print("####################freq_accusation:",freq_accusation,"freq_article:",freq_article,";num_copy:",num_copy)
            weight_accusation, weight_artilce=get_weight_freq_article(freq_accusation, freq_article)

        for k in range(num_copy):
            X.append(x)
            Y_accusation.append(y_accusation)
            Y_article.append(y_article)
            Y_deathpenalty.append(y_deathpenalty)
            Y_lifeimprisonment.append(y_lifeimprisonment)
            Y_imprisonment.append(float(imprisonment))
            weights_accusation.append(weight_accusation)
            weights_article.append(weight_artilce)

    #shuffle
    number_examples=len(X)
    X_=[]
    Y_accusation_=[]
    Y_article_=[]
    Y_deathpenalty_=[]
    Y_lifeimprisonment_=[]
    Y_imprisonment_=[]
    weights_accusation_=[]
    weights_article_=[]
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X_.append(X[index])
        Y_accusation_.append(Y_accusation[index])
        Y_article_.append(Y_article[index])
        Y_deathpenalty_.append(Y_deathpenalty[index])
        Y_lifeimprisonment_.append(Y_lifeimprisonment[index])
        Y_imprisonment_.append(Y_imprisonment[index])
        weights_accusation_.append(weights_accusation[index])
        weights_article_.append(weights_article[index])

    X_=np.array(X_)

    data = (X_, Y_accusation_, Y_article_, Y_deathpenalty_, Y_lifeimprisonment_, Y_imprisonment_,weights_accusation_,weights_article_)
    #dump to target file if and only if it is training data.
    print(data_type,"#########################finished.transform_data_to_index")
    if data_type == 'train':
        with open(target_file_path, 'ab') as target_file:# 'ab'
            print(data_type,"####################going to dump file:",target_file_path)
            pickle.dump(data, target_file,protocol=pickle.HIGHEST_PROTOCOL)
    else:
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

def transform_mulitihot_as_dense_list(multihot_list):
    length=len(multihot_list)
    result_list=[i for i in range(length) if multihot_list[i] > 0]
    return result_list


#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_or_load_vocabulary(data_path,predict_path,training_data_path,vocab_size,name_scope='cnn',test_mode=False,tokenize_style='word'):
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
           lines=lines[0:50000]
        #2.loop each line,put to counter
        c_inputs=Counter()
        c_accusation_labels=Counter()
        c_article_labels=Counter()
        for i,line in enumerate(lines):
            if i%10000==0:
                print(i)
            json_string = json.loads(line.strip())
            facts = json_string['fact']
            input_list = token_string_as_list(facts,tokenize_style=tokenize_style)
            if i % 10000 == 0:
                print("create_or_load_vocabulary:")
                print(input_list)
            c_inputs.update(input_list)

            accusation_list = json_string['meta']['accusation']
            c_accusation_labels.update(accusation_list)

            article_list = json_string['meta']['relevant_articles']
            c_article_labels.update(article_list)

        #3.get most frequency words
        vocab_list=c_inputs.most_common(vocab_size)
        word_vocab_file=predict_path+"/"+'word_freq.txt'
        if os.path.exists(word_vocab_file):
            print("word vocab file exists.going to delete it.")
            os.remove(word_vocab_file)
        word_freq_file=codecs.open(word_vocab_file,mode='a',encoding='utf-8')
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
    #string=string.decode("utf-8") #ADD TODO add this for python 2.7
    string=replace_money_value(string)  #TODO add normalize number ADD 2018.06.11
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt

def get_part_validation_data(valid,num_valid=6000*20):#6000
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment,weight_accusations,weight_artilces=valid
    number_examples=len(valid_X)
    permutation = np.random.permutation(number_examples)[0:num_valid]
    valid_X2, valid_Y_accusation2, valid_Y_article2, valid_Y_deathpenalty2, valid_Y_lifeimprisonment2, valid_Y_imprisonment2,weight_accusations2,weight_artilces=[],[],[],[],[],[],[],[]
    for index in permutation :
        valid_X2.append(valid_X[index])
        valid_Y_accusation2.append(valid_Y_accusation[index])
        valid_Y_article2.append(valid_Y_article[index])
        valid_Y_deathpenalty2.append(valid_Y_deathpenalty[index])
        valid_Y_lifeimprisonment2.append(valid_Y_lifeimprisonment[index])
        valid_Y_imprisonment2.append(valid_Y_imprisonment[index])
    return valid_X2,valid_Y_accusation2,valid_Y_article2,valid_Y_deathpenalty2,valid_Y_lifeimprisonment2,valid_Y_imprisonment2,weight_accusations2,weight_artilces


def load_accusation_articles_freq_dict(accusation_label2index,article_label2index,name_scope):
    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    #load dict of accusations
    accusation_freq_file = codecs.open(cache_vocabulary_label_pik + "/" + 'accusation_freq.txt', mode='r',encoding='utf-8')
    accusation_freq_lines=accusation_freq_file.readlines()
    accusation_freq_dict={}
    for i,line in enumerate(accusation_freq_lines):
        acc_label,freq=line.strip().split(splitter) #编造、故意传播虚假恐怖信息:122
        accusation_freq_dict[accusation_label2index[acc_label]]=int(freq)

    #load dict of articles
    article_freq_file = codecs.open(cache_vocabulary_label_pik + "/" + 'article_freq.txt', mode='r', encoding='utf-8')
    article_freq_lines=article_freq_file.readlines()
    article_freq_dict={}
    for i,line in enumerate(article_freq_lines):
        article_label,freq=line.strip().split(splitter) #397:3762
        article_freq_dict[article_label2index[int(article_label)]]=int(freq)
    return accusation_freq_dict,article_freq_dict


def get_weight_freq_article(freq_accusation,freq_article):
    if freq_accusation <= 100:
        weight_accusation = 3.0
    elif freq_accusation <= 200:
        weight_accusation = 2.0
    elif freq_accusation <= 500:
        weight_accusation = 1.5
    else:
        weight_accusation=1.0

    if freq_article <= 100:
        weight_artilce = 3.0
    elif freq_article <= 200:
        weight_artilce = 2.0
    elif freq_article <= 500:
        weight_artilce = 1.5
    else:
        weight_artilce=1.0
    return weight_accusation,weight_artilce


import re
def replace_money_value(string):
    #print("string:")
    #print(string)
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
#replace_money_value(x)

x="经审理查明，2012年上半年，被告人徐某使用蒋某提供的某某新农合本及身份证等证件，编造王某某甲亢性心脏病、脑溢血在中国人民解放军309医院的整套病历及住院费用证明，" \
  "并让桐柏县大河镇卫生院负责新农合报销的工作人员李某帮其办理新农合报销手续。从而徐某报销出新农合资金52459元。随后徐某分给蒋某某现金5000元。新农合报销资料显示王某某于" \
  "2012年6月2日至2012年7月16日在中国人民解放军第309医院以甲亢性心脏病、脑溢血住院，住院费用为95726.04元，新农合报销52459元。另查明，" \
  "被告人徐某因犯××于2015年4月9日被桐柏县人民法院判处××，并处罚金人民币××元，追缴违法所得3758.7元。刑期自2014年10月18日起至2020年10月17日止。" \
  "在河南省南阳市监狱服刑期间，发现徐某有上述漏罪。上述事实，被告人徐某在开庭审理过程中亦无异议，且有同案犯蒋某某的供述与辩解" \
  "，桐柏县新农合关于王某某的新农合报销病历及材料及中国解放军第309医院出具的证明等书证，到案证明，刑事判决书，准予解回罪犯的函，" \
  "被告人的常住人口基本信息等证据证实，足以认定。"
#x='XXXX同时从蒋2015年12月11日19时30分许某的支9号楼付宝账户中擅自转账61300余元，34534元，3599.34元，11400.123元,得到93443.454万元大幅度发，阿道夫12200元啊，得到3314342万元哦'
#result=replace_money_value(x)
#result2=jieba.lcut(result)
#for ss in result2:
#    print(ss)

#x='2018年9月7日到10号楼帮我拿1部价值5888元的手机，放到2单元，可以吗？'
#x="唐河县人民检察院指控：（一）××。2015年1月31日，被告人罗某在唐河县农33020元业银行营业厅帮不会取款操作的鹿某某取款，罗某趁机将鹿某某银行卡上的2900元存款转入自己账户后逃离。为指控上述犯罪事实，公诉机关当庭宣读和出示了被告人的供述、接处警登记表、银行卡交易笔录、被害人的陈述、现场勘查记录等相关证据。（二）××。2015年6月17日，被告人罗某尾随出售香囊的贾某某伺机作案，当晚10时许，贾某某行至唐河县泗洲宾馆后面的道内时，罗某将贾某某身上所挎提包抢走，致贾某某倒地，嘴部、腿部受伤。为指控上述犯罪事实，公诉机关当庭宣读和出示了被告人的供述、证人证言、被害人陈述、接处警登记表、现场勘查笔录等相关证据。综合上述指控，公诉机关认为，被告人罗某的行为已构成××、××，一人犯数罪，提请本院依据《中华人民共和国刑法》××、××、××之规定处罚。"
#result=replace_money_value(x)
#print("result2:",result)

#x='害人陈某1各项经济损失共计人民币35000元，并取得谅解'
#result=normalize_money(x)
#x='2某赔偿周某培各项损失30000元，取'
#result=normalize_money(x)
#x=' 同时从蒋某的支付宝账户中擅自转账1300余元。公'
#result=normalize_money(x)
#x=' 同时从蒋某的支付宝账户中擅自转账1300余元。公1400.123元'
#result=normalize_money(x)
#x='经鉴定，涉案轿车价值人民币11000元。'
#result=normalize_money(x)
#x="杭州市西湖区人民检察院指控，被告人黄某在大额债务无法归还、公司未正常经营的情况下，于2014年6月3日向被害人张某租赁宝马320i轿车一辆（价值人民币126000元）。后伪造张某的身份证、驾驶证，于同年6月6日，将车辆抵押给杭州德涵投资有限公司的吕营、王某，实际得款人民币100300元。被告人黄某将其中80000元用于归还债务，余款用于日常花销。被告人黄某的行为已构成××。对此指控，公诉机关当庭宣读和出示了证人证言、书证、鉴定意见等证据。"
#result=normalize_money(x)
#x='骗取他人财物共计11.98万元啊'
#result=normalize_money(x)
#x=''
#result=normalize_money(x)
#x=''
#training_data_path='../data/sample_multiple_label3.txt'
#vocab_size=100
#create_voabulary(training_data_path,vocab_size)
