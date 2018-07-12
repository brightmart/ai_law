import random
import json
import codecs

def get_label_dependency(data_path):
    """
    we compute co-exist rate between labels.
    :param data_path: 
    :return: list of list. length is label size. element of each sub list, it is (label_1,label_2,co_exisit_rate)
    """
    # 1.load data as lines
    train_file_object = codecs.open(data_path, mode='r', encoding='utf-8')
    lines = train_file_object.readlines()
    random.shuffle(lines)

    #2.loop each line, each is: label1's name, label2's name, how many times that co-exist,how many times that label_1 appears. e.g.  (label_1,label_2,numbers_co_exist,numbers_label_1)
    labe1label2_numbers_dict={}  # {label1_label2:1,}]
    label_dict={} # {label:number}
    for i,line in enumerate(lines):
        if len(line)<10: continue
        json_string = json.loads(line.strip())
        accusation_list = json_string['meta']['accusation']
        length=len(accusation_list)
        if length==2:#only check where exists two or more labels
            label1=accusation_list[0]
            label2=accusation_list[1]
            labe1label2_numbers_dict, label_dict=update_dict_list(labe1label2_numbers_dict,label_dict, label1, label2)
        elif length==3:
            label1=accusation_list[0]
            label2=accusation_list[1]
            label3=accusation_list[2]
            # from label1
            labe1label2_numbers_dict, label_dict=update_dict_list(labe1label2_numbers_dict, label_dict, label1, label2)
            # from label2
            labe1label2_numbers_dict, label_dict=update_dict_list(labe1label2_numbers_dict,label_dict, label1, label3)
            # from label3
            labe1label2_numbers_dict, label_dict=update_dict_list(labe1label2_numbers_dict,label_dict, label2, label3)

    # 3. compute co exist loop (label_1,label_2,numbers_co_exist/numbers_label_1)
    final_result={}
    ii=0
    for label1label2_string,number_co_existt in labe1label2_numbers_dict.items():
        label1,label2=label1label2_string.split("_")
        label_number=label_dict.get(label1,None)
        if label_number is None:continue
        rate=float(number_co_existt)/float(label_number)
        final_result[label1label2_string]=rate
        if rate>0.60 and number_co_existt>=3:
            print("NO:%s,label1:%s,label2:%s,rate:%f,label_number:%s" % (ii, label1, label2, rate,number_co_existt))
            ii=ii+1

def update_dict_list(labe1label2_numbers_dict,label_dict,label1,label2):
    # get co exist number
    number_co_exists=labe1label2_numbers_dict.get(label1 + "_" + label2)
    if number_co_exists is None:
        number_co_exists=1
    else:
        number_co_exists = labe1label2_numbers_dict.get(label1 + "_" + label2) + 1
    labe1label2_numbers_dict[label1 + "_" + label2] = number_co_exists

    # update label 1's dict
    label_number=label_dict.get(label1,None)
    if label_number is None:
        label_number=1
    else:
        label_number=label_number+1
    label_dict[label1]=label_number
    return labe1label2_numbers_dict,label_dict

data_path='../data_big/cail2018_big.json'
get_label_dependency(data_path)