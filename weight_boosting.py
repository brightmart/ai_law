# -*- coding: utf-8 -*-

def get_weight_for_batch(accusation_weight_dict,article_weight_dict,accusation_list,article_list):
    """
    get weight for a batch
    :param accusation_weight_dict: dict of accusation weight, key is accusation label, value is weight
    :param article_weight_dict: dict of article weight,key is article label,value is weight
    :param accusation_list: label list of accusation
    :param article_list: label list of article
    :return:
    """
    accusation_weight_list=[]
    article_weight_list=[]

    for aritlce,accusation in zip(accusation_list,article_list):
        pass
    return accusation_weight_list,article_weight_list in zip(accusation_list,article_list)

