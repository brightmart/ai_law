# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from HAN_model import HierarchicalAttention
from data_util import create_vocabulary,load_data_multilabel
import os
import word2vec
from sklearn.metrics import f1_score as f1_score_fn
from sklearn.metrics import r2_score #mean_squared_error
#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path","./data/data_train.json","path of traning data.")
tf.app.flags.DEFINE_string("valid_data_path","./data/data_valid.json","path of validation data.")
tf.app.flags.DEFINE_string("test_data_path","./data/data_test.json","path of validation data.")
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_float("keep_dropout_rate", 0.5, "percentage to keep when using dropout.") #0.65一次衰减多少

tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",400,"max sentence length") #400 TODO
tf.app.flags.DEFINE_integer("num_sentences",10,"number of sentences") #8 TODO
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")

tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("test_mode",False   ,"whether it is test mode. if it is test mode, only small percentage of data will be used")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #trainX, trainY, testX, testY = None, None, None, None

    vocab_word2index, accusation_label2index,articles_label2index= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope,test_mode=FLAGS.test_mode)
    deathpenalty_label2index={True:1,False:0}
    lifeimprisonment_label2index={True:1,False:0}
    vocab_size = len(vocab_word2index);print("cnn_model.vocab_size:",vocab_size);
    accusation_num_classes=len(accusation_label2index);article_num_classes=len(articles_label2index)
    deathpenalty_num_classes=len(deathpenalty_label2index);lifeimprisonment_num_classes=len(lifeimprisonment_label2index)
    print("accusation_num_classes:",accusation_num_classes);print("article_num_clasess:",article_num_classes)
    train,valid, test= load_data_multilabel(FLAGS.traning_data_path,FLAGS.valid_data_path,FLAGS.test_data_path,vocab_word2index, accusation_label2index,articles_label2index,deathpenalty_label2index,lifeimprisonment_label2index,
                                      FLAGS.sentence_len,name_scope=FLAGS.name_scope,test_mode=FLAGS.test_mode)
    train_X, train_Y_accusation, train_Y_article, train_Y_deathpenalty, train_Y_lifeimprisonment, train_Y_imprisonment = train
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment = valid
    test_X, test_Y_accusation, test_Y_article, test_Y_deathpenalty, test_Y_lifeimprisonment, test_Y_imprisonment = test
    #print some message for debug purpose
    print("length of training data:",len(train_X),";valid data:",len(valid_X),";test data:",len(test_X))
    print("trainX_[0]:", train_X[0]);
    train_Y_accusation_short = get_target_label_short(train_Y_accusation[0])
    train_Y_article_short = get_target_label_short(train_Y_article[0])
    print("train_Y_accusation_short:", train_Y_accusation_short,";train_Y_article_short:",train_Y_article_short)
    print("train_Y_deathpenalty:",train_Y_deathpenalty[0],";train_Y_lifeimprisonment:",train_Y_lifeimprisonment[0],";train_Y_imprisonment:",train_Y_imprisonment[0])
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model=HierarchicalAttention( accusation_num_classes,article_num_classes, deathpenalty_num_classes,lifeimprisonment_num_classes,FLAGS.learning_rate,FLAGS.batch_size,
                            FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.num_sentences,vocab_size, FLAGS.embed_size,FLAGS.hidden_size, FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            #for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(model.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                vocabulary_index2word={index:word for word,index in vocab_word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,FLAGS.word2vec_model_path)
        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(train_X)
        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",train_X[start:end],"train_X.shape:",train_X.shape)
                feed_dict = {model.input_x: train_X[start:end],model.input_y_accusation:train_Y_accusation[start:end],model.input_y_article:train_Y_article[start:end],
                             model.input_y_deathpenalty:train_Y_deathpenalty[start:end],model.input_y_lifeimprisonment:train_Y_lifeimprisonment[start:end],
                             model.input_y_imprisonment:train_Y_imprisonment[start:end],model.dropout_keep_prob: FLAGS.keep_dropout_rate}
                             #model.iter: iteration,model.tst: not FLAGS.is_training
                curr_loss,lr,_=sess.run([model.loss_val,model.learning_rate,model.train_op],feed_dict) #model.update_ema
                loss,counter=loss+curr_loss,counter+1
                if counter %100==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))
                ########################################################################################################
                if start!=0 and start%(300*FLAGS.batch_size)==0: # eval every 400 steps.
                    loss, f1_macro_accasation, f1_micro_accasation, f1_a_article, f1_i_aritcle, f1_a_death, f1_i_death, f1_a_life, f1_i_life, f1_r2_imprison = do_eval(sess, model, valid,iteration)
                    print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\tMacro_f1_article:%.3f\tMicro_f1_article:%.3f\tMacro_f1_deathpenalty%.3f\t"
                                "Micro_f1_deathpenalty%.3f\tMacro_f1_lifeimprisonment%.3f\tMicro_f1_lifeimprisonment%.3f\tR2_imprisonment:%.3f\t"
                                % (epoch, loss, f1_macro_accasation, f1_micro_accasation, f1_a_article, f1_i_aritcle,
                                   f1_a_death, f1_i_death, f1_a_life, f1_i_life, f1_r2_imprison))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                loss,f1_macro_accasation,f1_micro_accasation,f1_a_article,f1_i_aritcle,f1_a_death,f1_i_death,f1_a_life,f1_i_life,r2_imprison=do_eval(sess,model,valid,iteration)
                print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\tMacro_f1_article:%.3f\tMicro_f1_article:%.3f\tMacro_f1_deathpenalty%.3f\t"
                      "Micro_f1_deathpenalty%.3f\tMacro_f1_lifeimprisonment%.3f\tMicro_f1_lifeimprisonment%.3f\tR2_imprisonment:%.3f\t"
                      % (epoch,loss,f1_macro_accasation,f1_micro_accasation,f1_a_article,f1_i_aritcle,f1_a_death,f1_i_death,f1_a_life,f1_i_life,r2_imprison))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Testto 0.0
        #test_loss,macrof1,microf1 = do_eval(sess, textCNN, testX, testY,iteration)
        #print("Test Loss:%.3f\tMacro f1:%.3f\tMicro f1:%.3f" % (test_loss,macrof1,microf1))
        print("training completed...")
    pass


small_value=0.00001
def do_eval(sess,model,valid,iteration):
    valid_X, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment=valid
    number_examples=len(valid_X)
    print("number_examples:",number_examples)
    eval_loss,eval_counter=0.0,0
    batch_size=128 #TODO
    eval_macro_f1_accusation, eval_micro_f1_accusation,eval_r2_score_imprisonment,eval_macro_f1_article,eval_micro_f1_article,eval_r2_score_imprisonment = 0.0,0.0,0.0,0.0,0.0,0.0
    eval_macro_f1_deathpenalty,eval_micro_f1_deathpenalty,eval_macro_f1_lifeimprisonment,eval_micro_f1_lifeimprisonment=0.0,0.0,0.0,0.0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {model.input_x: valid_X[start:end],model.input_y_accusation:valid_Y_accusation[start:end],model.input_y_article:valid_Y_article[start:end],
                     model.input_y_deathpenalty:valid_Y_deathpenalty[start:end],model.input_y_lifeimprisonment:valid_Y_lifeimprisonment[start:end],
                     model.input_y_imprisonment:valid_Y_imprisonment[start:end],model.dropout_keep_prob: 1.0}#,model.iter: iteration,model.tst: True}
        curr_eval_loss, logits_accusation,logits_article,logits_deathpenalty,logits_lifeimprisonment,logits_imprisonment= sess.run(
            [model.loss_val,model.logits_accusation,model.logits_article,model.logits_deathpenalty,
             model.logits_lifeimprisonment,model.logits_imprisonment],feed_dict)#logits：[batch_size,label_size]

        macro_f1_accusation, micro_f1_accusation=get_f1_score(valid_Y_accusation[start:end],logits_accusation)
        macro_f1_article, micro_f1_article = get_f1_score(valid_Y_article[start:end],logits_article)
        macro_f1_deathpenalty, micro_f1_deathpenalty = get_f1_score(valid_Y_deathpenalty[start:end],logits_deathpenalty)
        macro_f1_lifeimprisonment, micro_f1_lifeimprisonment = get_f1_score(valid_Y_lifeimprisonment[start:end],logits_lifeimprisonment)
        r2_score_imprisonment=r2_score(valid_Y_imprisonment[start:end], logits_imprisonment)

        eval_loss=eval_loss+curr_eval_loss
        eval_macro_f1_accusation,eval_micro_f1_accusation=eval_macro_f1_accusation+macro_f1_accusation,eval_micro_f1_accusation+micro_f1_accusation
        eval_macro_f1_article,eval_micro_f1_article=eval_macro_f1_article+macro_f1_article,eval_micro_f1_article+micro_f1_article
        eval_macro_f1_deathpenalty,eval_micro_f1_deathpenalty=eval_macro_f1_deathpenalty+macro_f1_deathpenalty,eval_micro_f1_deathpenalty+micro_f1_deathpenalty
        eval_macro_f1_lifeimprisonment,eval_micro_f1_lifeimprisonment=eval_macro_f1_lifeimprisonment+macro_f1_lifeimprisonment,eval_micro_f1_lifeimprisonment+micro_f1_lifeimprisonment
        eval_r2_score_imprisonment=eval_r2_score_imprisonment+r2_score_imprisonment
        eval_counter=eval_counter+1

    return eval_loss/float(eval_counter*batch_size),float(eval_macro_f1_accusation)/float(eval_counter),float(eval_micro_f1_accusation)/float(eval_counter),\
           eval_macro_f1_article/float(eval_counter), eval_micro_f1_article/float(eval_counter),\
           eval_macro_f1_deathpenalty/float(eval_counter),eval_micro_f1_deathpenalty/float(eval_counter), \
           eval_macro_f1_lifeimprisonment/float(eval_counter), eval_micro_f1_lifeimprisonment/float(eval_counter), \
           eval_r2_score_imprisonment/float(eval_counter)

def get_f1_score(y_label,y_logit):
    label_target_accusation = transform_to_dense_label(y_label[0],only_one_flag=True)
    label_predict_accusation = [np.argmax(y_logit)]  # logits_accusation:[1,num_classes]-->logits_accusation[0]-->[num_classes]
    macro_f1 = f1_score_fn(label_target_accusation, label_predict_accusation, average='macro')  # y_true should like this:[0, 1, 2, 0, 1, 2]
    micro_f1 = f1_score_fn(label_target_accusation, label_predict_accusation, average='micro')
    return macro_f1,micro_f1

def transform_to_dense_label(label_list,only_one_flag=False):
    result_label=[]
    for i,l in enumerate(label_list):
        if l==1:
            result_label.append(i)
    if only_one_flag:
        result_label=[result_label[0]] #TODO TODO TODO TODO TODO
    return result_label

def compute_confuse_matrix(logit,predict):
    """
    compoute f1_score.
    :param logits: [label_size]
    :param evalY: [label_size]
    :return:
    """
    label=np.argmax(logit)
    true_positive=0  #TP:if label is true('1'), and predict is true('1')
    false_positive=0 #FP:if label is false('0'),but predict is ture('1')
    true_negative=0  #TN:if label is false('0'),and predict is false('0')
    false_negative=0 #FN:if label is false('0'),but predict is true('1')
    if predict==1 and label==1:
        true_positive=1
    elif predict==1 and label==0:
        false_positive=1
    elif predict==0 and label==0:
        true_negative=1
    elif predict==0 and label==1:
        false_negative=1

    return true_positive,false_positive,true_negative,false_negative

def compute_f1_score(label_list_top5,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0
    eval_y_short=get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/num_labels_predicted
    #R@5=Recall@5
    r_5=num_correct_label/all_real_labels
    f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
    return f1_score,p_5,r_5

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
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
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()