# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import tensorflow as tf
import numpy as np
from .HAN_model import HierarchicalAttention
from .data_util_test import token_string_as_list,imprisonment_mean,imprisonment_std,UNK_ID,load_word_vocab,load_label_dict_accu,load_label_dict_article,pad_truncate_list

class Predictor(object):
    def __init__(self):
        """
        init method required. set batch_size, and load some resources.
        """
        self.batch_size =1024


        FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_string("ckpt_dir", "predictor/checkpoint_big/", "checkpoint location for the model")
        tf.app.flags.DEFINE_string("vocab_word_path", "predictor/word_freq.txt", "path of word vocabulary.")
        tf.app.flags.DEFINE_string("accusation_label_path", "predictor/accu.txt", "path of accusation labels.")
        tf.app.flags.DEFINE_string("article_label_path", "predictor/law.txt", "path of law labels.")

        tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
        tf.app.flags.DEFINE_integer("decay_steps", 1000,"how many steps before decay learning rate.")
        tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
        tf.app.flags.DEFINE_integer("sentence_len", 400, "max sentence length")
        tf.app.flags.DEFINE_integer("num_sentences", 16, "number of sentences")
        tf.app.flags.DEFINE_integer("embed_size", 256, "embedding size") #64
        tf.app.flags.DEFINE_integer("hidden_size", 256, "hidden size")  #128
        tf.app.flags.DEFINE_integer("num_filters", 256, "number of filter for a filter map used in CNN.") #128
        tf.app.flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")
        tf.app.flags.DEFINE_string("model", "text_cnn", "name of model:han,c_gru,c_gru2,gru,text_cnn")
        #tf.app.flags.DEFINE_boolean("is_training_flag", False, "is traning.true:tranining,false:testing/inference")

        filter_sizes = [2,3,4,5]#,6,7,8]#[2,3,4,5]#[6, 7, 8, 9, 10]  # [30,40,50] #8
        stride_length = 1

        #1.load label dict, restore model from checkpoint
        # 1.load label dict
        self.vocab_word2index=load_word_vocab(FLAGS.vocab_word_path)
        accusation_label2index=load_label_dict_accu(FLAGS.accusation_label_path)
        articles_label2index=load_label_dict_article(FLAGS.article_label_path)

        deathpenalty_label2index = {True: 1, False: 0}
        lifeimprisonment_label2index = {True: 1, False: 0}
        vocab_size = len(self.vocab_word2index);
        accusation_num_classes = len(accusation_label2index);
        article_num_classes = len(articles_label2index)
        deathpenalty_num_classes = len(deathpenalty_label2index);
        lifeimprisonment_num_classes = len(lifeimprisonment_label2index)

        # 2.restore checkpoint
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess=tf.Session(config=config)
        self.model = HierarchicalAttention(accusation_num_classes, article_num_classes, deathpenalty_num_classes,lifeimprisonment_num_classes, FLAGS.learning_rate, self.batch_size,FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.num_sentences,vocab_size, FLAGS.embed_size, FLAGS.hidden_size
                                 ,num_filters = FLAGS.num_filters, model = FLAGS.model, filter_sizes = filter_sizes, stride_length = stride_length)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        self.sess=sess
        self.FLAGS=FLAGS

    def predict_with_model_batch(self,contents):
        """
        predict result use model
        :param contents:  a list. each element is a string,which represent of fact of law case.
        :return: a dict

        """
        model=self.model
        input_X=[]
        #1.get fact, 1)tokenize,2)word to index, 3)pad &truncate
        length_contents=len(contents)
        #################################################
        contents_padded=[]
    #if length_contents<self.batch_size:
        for i in range(self.batch_size):
            if i<length_contents:
                contents_padded.append(contents[i])
            else:
                #print(str(i),".going to padd")
                contents_padded.append(contents[0]) #pad the list to batch_size,
        #################################################

        for i,fact in enumerate(contents_padded):
            input_list = token_string_as_list(fact)  # tokenize
            x = [self.vocab_word2index.get(x, UNK_ID) for x in input_list]  # transform input to index
            x = pad_truncate_list(x, self.FLAGS.sentence_len, value=0.,truncating='pre')  # padding to max length.remove sequence that longer than max length from beginning.
            input_X.append(x)
        #2.feed data and get logit
        feed_dict = {model.input_x: input_X,model.dropout_keep_prob: 1.0,model.is_training_flag:False}
        logits_accusations,logits_articles,logits_deathpenaltys,logits_lifeimprisonments,logits_imprisonments= self.sess.run([model.logits_accusation,model.logits_article,model.logits_deathpenalty,model.logits_lifeimprisonment,model.logits_imprisonment],feed_dict)
        #logits_accusation:[batch_size,num_classes]-->logits[0].shape:[num_classes];logits_imprisonment:[batch_size]

        #3.get label_index
        #print("=====>logits_accusations.shape:",logits_accusations.shape) # (20, 202)
        #batch_size=len(contents)
        result_list=[]
        for i in range(length_contents):
            logits_accusation=logits_accusations[i] #TODO since batch_size=1, we only need to check first element.
            accusations_predicted= [j+1 for j in range(len(logits_accusation)) if logits_accusation[j]>=0.5]  #TODO ADD ONE e.g.[2,12,13,10]
            if len(accusations_predicted)<1:
                accusations_predicted=[np.argmax(logits_accusation)+1] #TODO ADD ONE
            logits_article=logits_articles[i]
            articles_predicted= [j+1 for j in range(len(logits_article)) if logits_article[j]>=0.5]  ##TODO ADD ONE e.g.[2,12,13,10]
            if len(articles_predicted)<1:
                articles_predicted=[np.argmax(logits_article)+1] #TODO ADD ONE
            deathpenalty_predicted=np.argmax(logits_deathpenaltys[i]) #0 or 1
            lifeimprisonment_predicted=np.argmax(logits_lifeimprisonments[i]) #0 or 1
            #print("=====>logits_imprisonments[i]:",logits_imprisonments[i])
            imprisonment_predicted=int(round(logits_imprisonments[i])) #*imprisonment_std)
            imprisonment=0
            if deathpenalty_predicted==1:
                imprisonment=-2
            elif lifeimprisonment_predicted==1:
                imprisonment=-1
            else:
                imprisonment=imprisonment_predicted
            dictt={}
            dictt['accusation']=accusations_predicted
            dictt['articles'] =articles_predicted
            dictt['imprisonment'] =imprisonment
            result_list.append(dictt)
        #print("accusation_predicted:",accusations_predicted,";articles_predicted:",articles_predicted,";deathpenalty_predicted:",deathpenalty_predicted,";lifeimprisonment_predicted:",
        #      lifeimprisonment_predicted,";imprisonment_predicted:",imprisonment_predicted,";imprisonment:",imprisonment)

        #4.return
        return result_list


    def predict(self, contents): #get facts, use model to make a prediction.
        """
        standard predict method required.
        :param content:  a list. each element is a string,which represent of fact of law case.
        :return: a dict
        """
        result_list=self.predict_with_model_batch(contents)
        return result_list
#predict=Predictor()
#content1=u"酒泉市肃州区人民检察院指控，1、2015年5月15日18时许，被告人孙某窜至酒泉市肃州区南大街蚂蚁服装店内，趁被害人沈某某不备，盗窃沈某某装在上衣口袋内的白色三星7109型手机一部，后将该手机以200元价格出售给一陌生男子。该手机价值1500元；2、2015年5月16日22时许，被告人孙某窜至酒泉市肃州区大明步行街贝某地婴童坊内，趁被害人马某某不备，盗窃马某某放在吧台旁的黄色女式包内的绿色钱包一个。钱包内装有人民币4000元、身份证、银行卡等物品，损失价值合计4000元。"
#content2=u"成安县人民检察院指控，2013年2月11日15时许，犯罪嫌疑人吴某因与本村村民任某甲闹离婚发生矛盾，吴某伙同其亲戚到任某甲家对任某甲进行殴打，并在任某甲父亲任某丁外出回家时，与任某丁打架，在打架过程中，吴某用刀将任某丁左脸及左某划伤，后经法医鉴定，任某丁的伤已构成轻伤。公诉机关提供了相应的证据证明上述事实，要求本院依照《中华人民共和国刑法》第二百三十四之规定，以××追究被告人吴某的刑事责任。"
#content3=u"公诉机关指控：2013年11月7日11时许，被告人钟某某在佛山市高明区明城镇某某金属有限公司工作时，趁其他员工去吃午饭时将公司的19条黄铜锭从公司厂房墙角的洞口丢到外面，并于2013年11月8日2时许独自驾驶一辆女装摩托车到公司厂房墙外将其中的15条黄铜锭盗走并销赃给他人。2013年11月8日11时许，被告人钟某某以同样的方式将11条黄铜锭丢到公司厂房外面，并于2013年11月10日2时许独自驾驶一辆女装摩托车准备把剩余的15条黄铜锭盗走，行至公司厂房墙外时被伏击的公安人员抓获。公安人员扣押了上述15条黄铜锭，并已发还给被害人。经价格鉴定，被盗的30条黄铜锭共价值人民币13884元。公诉机关认为被告人钟某某无视国家法律，以非法占有为目的，盗窃他人财物，价值人民币13884元，数额较大，其行为已触犯了《中华人民共和国刑法》××之规定，应以××追究其刑事责任。在法庭上，公诉机关还详细阐述了关于指控被告人盗窃数额为13884元的相关依据和计算方法。公诉机关建议对被告人钟某某判处六个月至一年六个月××，并处罚金。提请本院依法判处。"
#content_list=[content1,content2,content3]
#result=predict.predict(content_list)
#print("result:",result)