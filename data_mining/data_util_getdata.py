import codecs
import random

def xxx(file_path,target_path):
    #1.read file
    file_object = codecs.open(file_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    length = len(lines)
    #2.shuffle
    random.shuffle(lines)
    #3.get some
    five_percentage = int(length * 0.10)
    lines=lines[0:five_percentage]
    #4.save
    target_object = codecs.open(target_path, mode='a', encoding='utf-8')
    for i, line in enumerate(lines):
        target_object.write(line+"\n")
    target_object.close()
    file_object.close()


file_path='../data_big/cail2018_big.json'
target_path='../data_big/cail2018_big_small.json'
xxx(file_path,target_path)