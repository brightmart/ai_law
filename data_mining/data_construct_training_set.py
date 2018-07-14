# -*- coding: utf-8 -*-
import random
import json
import codecs

max_samples=13000
dict_downsampling_accusation={u'盗窃':float(max_samples)/float(373122),
                              u'危险驾驶':float(max_samples)/float(336191),
                              u'故意伤害':float(max_samples)/float(194762),
                              u'交通肇事':float(max_samples)/float(162755),
                              u'走私、贩卖、运输、制造毒品':float(max_samples)/float(123320),
                              u'容留他人吸毒':float(max_samples)/float(59201),
                              u'诈骗': float(max_samples) / float(54461),
                              u'寻衅滋事': float(max_samples) / float(35331),
                              u'抢劫': float(max_samples) / float(26211),
                              u'信用卡诈骗': float(max_samples) / float(22217),
                              u'开设赌场': float(max_samples) / float(19547),
                              u'非法持有、私藏枪支、弹药': float(max_samples) / float(19270),
                              u'非法持有毒品': float(max_samples) / float(18912),
                              u'妨害公务': float(max_samples) / float(18764)
                              }
def down_sampling(train_lines):
    """
    down sampling: remove some lines
    :param train_lines:
    :return:
    """
    ignore_count=0
    train_lines_return=[]
    for i,line in enumerate(train_lines):
        json_string = json.loads(line.strip())
        accusation_list = json_string['meta']['accusation']
        article_list = json_string['meta']['relevant_articles']
        accusation = accusation_list[0]
        if len(accusation_list)==1 and len(article_list)==1 and  accusation in dict_downsampling_accusation: # only drop some simple sample.
            # has a possibility to put to keep list.
            keep_rate=dict_downsampling_accusation[accusation]
            if random.random()<keep_rate:
                train_lines_return.append(line)
            else:
                if i%1000==0:print("going to ignore:");print(line)
                ignore_count=ignore_count+1
        else:
            train_lines_return.append(line)
    return train_lines_return,ignore_count


def down_sampling_data(soure_file,target_file):
    """
    down sampling: read-down sampling-write
    :param soure_file:
    :param target_file:
    :return:
    """
    #1. read data, and put to list
    train_file_object = codecs.open(soure_file, mode='r', encoding='utf-8')
    target_object = codecs.open(target_file, mode='a', encoding='utf-8')

    train_lines_original = train_file_object.readlines()
    random.shuffle(train_lines_original)
    #2. down_sampling
    print("lines1:",len(train_lines_original))
    train_lines_original,ignore_count=down_sampling(train_lines_original)
    print("lines2:",len(train_lines_original),";ignore_count:",ignore_count)
    #3. save to a new file
    for i,line in enumerate(train_lines_original):
        target_object.write(line)
    target_object.close()
    train_file_object.close()

soure_file='../data_big/cail2018_big.json' #1710856
target_file='../data_big/cail2018_big_downsmapled.json' #590083
down_sampling_data(soure_file,target_file)