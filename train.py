# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from predictor.model import HierarchicalAttention
from data_util import create_or_load_vocabulary,load_data_multilabel,get_part_validation_data #,imprisonment_mean,imprisonment_std
import os
from evaluation_matrix import *
import gensim
from gensim.models import KeyedVectors
#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path","./data","path of traning data.")
tf.app.flags.DEFINE_string("traning_data_file","./data_big/cail2018_big.json","path of traning data.") #./data/data_train.json
tf.app.flags.DEFINE_string("valid_data_file","./data/data_valid.json","path of validation data.")
tf.app.flags.DEFINE_string("test_data_path","./data/data_test.json","path of validation data.")
tf.app.flags.DEFINE_string("predict_path","./predictor","path of traning data.")
tf.app.flags.DEFINE_string("ckpt_dir","./predictor/checkpoint/","checkpoint location for the model") #save to here, so make it easy to upload for test

tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.") #80000
tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate") #0.001
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_float("keep_dropout_rate", 0.5, "percentage to keep when using dropout.") #0.65一次衰减多少
tf.app.flags.DEFINE_integer("sentence_len",400,"max sentence length")
tf.app.flags.DEFINE_integer("num_sentences",16,"number of sentences")
tf.app.flags.DEFINE_integer("embed_size",64,"embedding size") #300-->64
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size") #128
tf.app.flags.DEFINE_integer("num_filters",128,"number of filter for a filter map used in CNN.") #128

tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","data/data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin","word2vec's vocabulary and vectors") # data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5--->data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin
#tf.app.flags.DEFINE_string("word2vec_model_path","data_big/law_embedding_64_skipgram.bin","word2vec's vocabulary and vectors")
#tf.app.flags.DEFINE_string("name_scope","dp_cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("test_mode",False,"whether it is test mode. if it is test mode, only small percentage of data will be used")

tf.app.flags.DEFINE_string("model","dp_cnn","name of model:han,text_cnn,dp_cnn,c_gru,c_gru2,gru,pooling")
tf.app.flags.DEFINE_string("pooling_strategy","hier","pooling strategy used when model is pooling. {avg,max,concat,hier}")
#you can change this
filter_sizes=[2,3,4,5] #,6,7,8]# [6, 7, 8, 9, 10]

stride_length=1
def main(_):
    print("model:",FLAGS.model)
    name_scope=FLAGS.model
    vocab_word2index, accusation_label2index,articles_label2index= create_or_load_vocabulary(FLAGS.data_path,FLAGS.predict_path,FLAGS.traning_data_file,FLAGS.vocab_size,name_scope=name_scope,test_mode=FLAGS.test_mode)
    deathpenalty_label2index={True:1,False:0}
    lifeimprisonment_label2index={True:1,False:0}
    vocab_size = len(vocab_word2index);print("cnn_model.vocab_size:",vocab_size);
    accusation_num_classes=len(accusation_label2index);article_num_classes=len(articles_label2index)
    deathpenalty_num_classes=len(deathpenalty_label2index);lifeimprisonment_num_classes=len(lifeimprisonment_label2index)
    print("accusation_num_classes:",accusation_num_classes);print("article_num_clasess:",article_num_classes)
    train,valid, test= load_data_multilabel(FLAGS.traning_data_file,FLAGS.valid_data_file,FLAGS.test_data_path,vocab_word2index, accusation_label2index,articles_label2index,deathpenalty_label2index,lifeimprisonment_label2index,
                                      FLAGS.sentence_len,name_scope=name_scope,test_mode=FLAGS.test_mode)
    train_X, train_Y_accusation, train_Y_article, train_Y_deathpenalty, train_Y_lifeimprisonment, train_Y_imprisonment,train_weights_accusation,train_weights_article = train
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment,valid_weights_accusation,valid_weights_article = valid
    test_X, test_Y_accusation, test_Y_article, test_Y_deathpenalty, test_Y_lifeimprisonment, test_Y_imprisonment,test_weights_accusation,test_weights_article = test
    #print some message for debug purpose
    print("length of training data:",len(train_X),";valid data:",len(valid_X),";test data:",len(test_X))
    print("trainX_[0]:", train_X[0]);
    train_Y_accusation_short1 = get_target_label_short(train_Y_accusation[0]);train_Y_accusation_short2 = get_target_label_short(train_Y_accusation[1]);train_Y_accusation_short3 = get_target_label_short(train_Y_accusation[2]);train_Y_accusation_short4 = get_target_label_short(train_Y_accusation[20]);train_Y_accusation_short5 = get_target_label_short(train_Y_accusation[200])
    train_Y_article_short = get_target_label_short(train_Y_article[0])
    print("train_Y_accusation_short:", train_Y_accusation_short1,train_Y_accusation_short2,train_Y_accusation_short3,train_Y_accusation_short4,train_Y_accusation_short4,";train_Y_article_short:",train_Y_article_short)
    print("train_Y_deathpenalty:",train_Y_deathpenalty[0],";train_Y_lifeimprisonment:",train_Y_lifeimprisonment[0],";train_Y_imprisonment:",train_Y_imprisonment[0])
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model=HierarchicalAttention( accusation_num_classes,article_num_classes, deathpenalty_num_classes,lifeimprisonment_num_classes,FLAGS.learning_rate,FLAGS.batch_size,
                            FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.num_sentences,vocab_size, FLAGS.embed_size,FLAGS.hidden_size,
                                     num_filters=FLAGS.num_filters,model=FLAGS.model,filter_sizes=filter_sizes,stride_length=stride_length,pooling_strategy=FLAGS.pooling_strategy)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            for i in range(2): #decay learning rate if necessary.
                print(i,"Going to decay learning rate by half.")
                sess.run(model.learning_rate_decay_half_op)
                #sess.run(model.learning_rate_decay_half_op)

        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_pretrained_embedding: #load pre-trained word embedding
                vocabulary_index2word={index:word for word,index in vocab_word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,FLAGS.word2vec_model_path,model.Embedding)
                #assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,FLAGS.word2vec_model_path2,model.Embedding2) #TODO

        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(train_X)
        batch_size=FLAGS.batch_size
        iteration=0
        accasation_score_best=-100


        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss_total, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",train_X[start:end],"train_X.shape:",train_X.shape)
                feed_dict = {model.input_x: train_X[start:end],model.input_y_accusation:train_Y_accusation[start:end],model.input_y_article:train_Y_article[start:end],
                             model.input_y_deathpenalty:train_Y_deathpenalty[start:end],model.input_y_lifeimprisonment:train_Y_lifeimprisonment[start:end],
                             model.input_y_imprisonment:train_Y_imprisonment[start:end],model.input_weight_accusation:train_weights_accusation[start:end],
                             model.input_weight_article:train_weights_article[start:end],model.dropout_keep_prob: FLAGS.keep_dropout_rate,
                             model.is_training_flag:FLAGS.is_training_flag}
                             #model.iter: iteration,model.tst: not FLAGS.is_training
                current_loss,lr,loss_accusation,loss_article,loss_deathpenalty,loss_lifeimprisonment,loss_imprisonment,l2_loss,_=\
                    sess.run([model.loss_val,model.learning_rate,model.loss_accusation,model.loss_article,model.loss_deathpenalty,
                                         model.loss_lifeimprisonment,model.loss_imprisonment,model.l2_loss,model.train_op],feed_dict) #model.update_ema
                loss_total,counter=loss_total+current_loss,counter+1
                if counter %20==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,float(loss_total)/float(counter),lr))
                if counter %60==0:
                    print("Loss_accusation:%.3f\tLoss_article:%.3f\tLoss_deathpenalty:%.3f\tLoss_lifeimprisonment:%.3f\tLoss_imprisonment:%.3f\tL2_loss:%.3f\tCurrent_loss:%.3f\t"
                          %(loss_accusation,loss_article,loss_deathpenalty,loss_lifeimprisonment,loss_imprisonment,l2_loss,current_loss))
                ########################################################################################################
                if start!=0 and start%(3000*FLAGS.batch_size)==0: # eval every 400 steps.
                    loss, f1_macro_accasation, f1_micro_accasation, f1_a_article, f1_i_aritcle, f1_a_death, f1_i_death, f1_a_life, f1_i_life, score_penalty = \
                        do_eval(sess, model, valid,iteration,accusation_num_classes,article_num_classes,accusation_label2index)
                    accasation_score=((f1_macro_accasation+f1_micro_accasation)/2.0)*100.0
                    article_score=((f1_a_article+f1_i_aritcle)/2.0)*100.0
                    score_all=accasation_score+article_score+score_penalty #3ecfDzJbjUvZPUdS
                    print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\tMacro_f1_article:%.3f Micro_f1_article:%.3f Macro_f1_deathpenalty:%.3f\t"
                                "Micro_f1_deathpenalty:%.3f\tMacro_f1_lifeimprisonment:%.3f\tMicro_f1_lifeimprisonment:%.3f\t"
                                % (epoch, loss, f1_macro_accasation, f1_micro_accasation, f1_a_article, f1_i_aritcle,f1_a_death, f1_i_death, f1_a_life, f1_i_life))
                    print("1.Accasation Score:", accasation_score, ";2.Article Score:", article_score, ";3.Penalty Score:",score_penalty, ";Score ALL:", score_all)
                    # save model to checkpoint
                    if accasation_score>accasation_score_best:
                        save_path = FLAGS.ckpt_dir + "model.ckpt" #TODO temp remove==>only save checkpoint for each epoch once.
                        print("going to save check point.")
                        saver.save(sess, save_path, global_step=epoch)
                        accasation_score_best=accasation_score
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                loss,f1_macro_accasation,f1_micro_accasation,f1_a_article,f1_i_aritcle,f1_a_death,f1_i_death,f1_a_life,f1_i_life,score_penalty=\
                    do_eval(sess,model,valid,iteration,accusation_num_classes,article_num_classes,accusation_label2index)
                accasation_score = ((f1_macro_accasation + f1_micro_accasation) / 2.0) * 100.0
                article_score = ((f1_a_article + f1_i_aritcle) / 2.0) * 100.0
                score_all = accasation_score + article_score + score_penalty
                print()
                print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\tMacro_f1_article:%.3f\tMicro_f1_article:%.3f\tMacro_f1_deathpenalty:%.3f\t"
                      "Micro_f1_deathpenalty:%.3f\tMacro_f1_lifeimprisonment:%.3f\tMicro_f1_lifeimprisonment:%.3f\t"
                      % (epoch,loss,f1_macro_accasation,f1_micro_accasation,f1_a_article,f1_i_aritcle,f1_a_death,f1_i_death,f1_a_life,f1_i_life))
                print("===>1.Accasation Score:", accasation_score, ";2.Article Score:", article_score,";3.Penalty Score:",score_penalty,";Score ALL:",score_all)

                #save model to checkpoint
                if accasation_score > accasation_score_best:
                    save_path=FLAGS.ckpt_dir+"model.ckpt"
                    print("going to save check point.")
                    saver.save(sess,save_path,global_step=epoch)
                    accasation_score_best = accasation_score
            #if (epoch == 2 or epoch == 4 or epoch == 7 or epoch==10 or epoch == 13  or epoch==19):
            #if (epoch == 1 or epoch == 3 or epoch == 6 or epoch == 9 or epoch == 12 or epoch == 18):
            if (epoch == 0 or epoch == 2 or epoch == 4 or epoch == 6 or epoch == 9 or epoch == 13):

                for i in range(2):
                    print(i, "Going to decay learning rate by half.")
                    sess.run(model.learning_rate_decay_half_op)

        # 5.最后在测试集上做测试，并报告测试准确率 Testto 0.0
        loss_test, f1_macro_accasation_test, f1_micro_accasation_test, f1_a_article_test, f1_i_aritcle_test, f1_a_death_test, f1_i_death_test, f1_a_life_test, f1_i_life_test, score_penalty_test=\
            do_eval(sess, model, test, iteration, accusation_num_classes, article_num_classes, accusation_label2index)
        print("TEST.FINAL.Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\tMacro_f1_article:%.3f\tMicro_f1_article:%.3f\tMacro_f1_deathpenalty:%.3f\t"
                    "Micro_f1_deathpenalty:%.3f\tMacro_f1_lifeimprisonment:%.3f\tMicro_f1_lifeimprisonment:%.3f\t"
                    % (epoch, loss_test, f1_macro_accasation_test, f1_micro_accasation_test, f1_a_article_test, f1_i_aritcle_test, f1_a_death_test,
                       f1_i_death_test, f1_a_life_test, f1_i_life_test))
        accasation_score_test = ((f1_macro_accasation_test + f1_micro_accasation_test) / 2.0) * 100.0
        article_score_test = ((f1_a_article_test + f1_i_aritcle_test) / 2.0) * 100.0
        score_all_test = accasation_score_test + article_score_test + score_penalty_test
        print("TEST.Accasation Score:", accasation_score_test, ";2.Article Score:", article_score_test, ";3.Penalty Score:",score_penalty_test, ";Score ALL:", score_all_test)

        #print("Test Loss:%.3f\tMacro f1:%.3f\tMicro f1:%.3f" % (test_loss,macrof1,microf1))
        print("training completed...")
    pass

def do_eval(sess,model,valid,iteration,accusation_num_classes,article_num_classes,accusation_label2index):
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment,_,_=get_part_validation_data(valid)
    number_examples=len(valid_X)
    print("number_examples:",number_examples)
    eval_loss,eval_counter=0.0,0
    batch_size=FLAGS.batch_size
    label_dict_accusation=init_label_dict(accusation_num_classes)
    label_dict_article=init_label_dict(article_num_classes)
    label_dict_deathpenalty = init_label_dict(2)
    label_dict_lifeimprisonment = init_label_dict(2)

    eval_macro_f1_accusation, eval_micro_f1_accusation,eval_r2_score_imprisonment,eval_macro_f1_article,eval_micro_f1_article,eval_r2_score_imprisonment = 0.0,0.0,0.0,0.0,0.0,0.0
    eval_penalty_score=0.0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {model.input_x: valid_X[start:end],
                     model.input_y_accusation:valid_Y_accusation[start:end],model.input_y_article:valid_Y_article[start:end],
                     model.input_y_deathpenalty:valid_Y_deathpenalty[start:end],model.input_y_lifeimprisonment:valid_Y_lifeimprisonment[start:end],
                     model.input_y_imprisonment:valid_Y_imprisonment[start:end],model.input_weight_accusation:
                         [1.0 for i in range(batch_size)],model.input_weight_article:[1.0 for i in range(batch_size)],
                     model.dropout_keep_prob: 1.0,model.is_training_flag:False}#,model.iter: iteration,model.tst: True}
        curr_eval_loss, logits_accusation,logits_article,logits_deathpenalty,logits_lifeimprisonment,logits_imprisonment= sess.run(
                        [model.loss_val,model.logits_accusation,model.logits_article,model.logits_deathpenalty,model.logits_lifeimprisonment,model.logits_imprisonment],feed_dict)#logits：[batch_size,label_size]
        #compute confuse matrix for accusation,relevant article,death penalty,life imprisonment
        label_dict_accusation=compute_confuse_matrix_batch(valid_Y_accusation[start:end],logits_accusation,label_dict_accusation,name='accusation')
        label_dict_article = compute_confuse_matrix_batch(valid_Y_article[start:end],logits_article,label_dict_article,name='article')
        label_dict_deathpenalty = compute_confuse_matrix_batch(valid_Y_deathpenalty[start:end],logits_deathpenalty,label_dict_deathpenalty,name='deathpenalty')
        label_dict_lifeimprisonment = compute_confuse_matrix_batch(valid_Y_lifeimprisonment[start:end],logits_lifeimprisonment,label_dict_lifeimprisonment,name='lifeimprisionment')
        penalty_score=compute_penalty_score_batch(valid_Y_deathpenalty[start:end], logits_deathpenalty,
                                                    valid_Y_lifeimprisonment[start:end], logits_lifeimprisonment,valid_Y_imprisonment, logits_imprisonment)
        eval_penalty_score=eval_penalty_score+penalty_score
        eval_loss=eval_loss+curr_eval_loss
        eval_counter=eval_counter+1

    #compute f1_micro & f1_macro for accusation,article,deathpenalty,lifeimprisonment
    f1_micro_accusation,f1_macro_accusation=compute_micro_macro(label_dict_accusation) #label_dict_accusation is a dict, key is: accusation,value is: (TP,FP,FN). where TP is number of True Positive
    compute_accusation_f1_score_write_for_debug(label_dict_accusation, accusation_label2index)

    f1_micro_article, f1_macro_article = compute_micro_macro(label_dict_article)
    f1_micro_deathpenalty, f1_macro_deathpenalty = compute_micro_macro(label_dict_deathpenalty)
    f1_micro_lifeimprisonment, f1_macro_lifeimprisonment = compute_micro_macro(label_dict_lifeimprisonment)
    print("f1_micro_accusation:",f1_micro_accusation,";f1_macro_accusation:",f1_macro_accusation)
    return eval_loss/float(eval_counter+small_value),f1_macro_accusation,f1_micro_accusation, f1_macro_article, f1_micro_article, \
           f1_macro_deathpenalty, f1_micro_deathpenalty,f1_macro_lifeimprisonment, f1_micro_lifeimprisonment,eval_penalty_score/float(eval_counter+small_value)


def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path,embedding_instance):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    ##word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    binary_flag = True
    if '.bin' not in word2vec_model_path:
        binary_flag = False
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary_flag,unicode_errors='ignore')
    #word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, unicode_errors='ignore')  #
    word2vec_dict = {}
    count_=0
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        if count_==0:
            print("pretrained word embedding size:",str(len(vector)))
            count_=count_+1
        if '.bin' not in word2vec_model_path:
            word2vec_dict[word] = vector
        else:
            word2vec_dict[word] = vector /np.linalg.norm(vector) # normalize vector only when word2vec data is a .bin file
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(3.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(embedding_instance,word_embedding)  #TODO model.Embedding. assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("====>>>>word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()