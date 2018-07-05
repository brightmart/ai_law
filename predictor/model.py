# -*- coding: utf-8 -*-
# HierarchicalAttention: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier. 2017-06-13
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib import rnn

class HierarchicalAttention:
    def __init__(self,  accusation_num_classes,article_num_classes, deathpenalty_num_classes,lifeimprisonment_num_classes,learning_rate,
                        batch_size, decay_steps, decay_rate, sequence_length, num_sentences,vocab_size, embed_size,hidden_size,
                        initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=1.0,max_pooling_style='max_pooling',
                        model='c_gru',num_filters=128,filter_sizes=[8],stride_length=4,pooling_strategy='hier',hpcnn_filter_size=3,hpcnn_number_filters=64,num_repeat=4):#hpcnn_number_filters=32
        """init all hyperparameter here"""
        # set hyperparamter
        self.accusation_num_classes = accusation_num_classes
        self.article_num_classes=article_num_classes
        self.deathpenalty_num_classes=deathpenalty_num_classes
        self.lifeimprisonment_num_classes=lifeimprisonment_num_classes
        self.batch_size = batch_size
        self.total_sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.6)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.max_pooling_style=max_pooling_style
        self.model=model
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.stride_length=stride_length
        self.pooling_strategy=pooling_strategy
        self.hpcnn_filter_size=hpcnn_filter_size
        self.hpcnn_number_filters=hpcnn_number_filters
        self.num_repeat=num_repeat
        print("self.filter_sizes:",self.filter_sizes,";self.num_filters:",self.num_filters,"; self.stride_length:", self.stride_length)

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [batch_size, self.total_sequence_length], name="input_x")
        self.input_y_accusation=tf.placeholder(tf.float32, [batch_size, self.accusation_num_classes],name="input_y_accusation")
        self.input_y_article = tf.placeholder(tf.float32, [batch_size, self.article_num_classes],name="input_y_article")
        self.input_y_deathpenalty = tf.placeholder(tf.float32, [batch_size, self.deathpenalty_num_classes], name="input_y_deathpenalty")
        self.input_y_lifeimprisonment = tf.placeholder(tf.float32, [batch_size, self.lifeimprisonment_num_classes], name="input_y_lifeimprisonment")
        self.input_y_imprisonment = tf.placeholder(tf.float32, [batch_size], name="input_y_imprisonment")
        self.input_weight_accusation = tf.placeholder(tf.float32, [batch_size], name="input_weight_accusation")
        self.input_weight_article = tf.placeholder(tf.float32, [batch_size], name="input_weight_article")

        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        self.sequence_length = int(self.total_sequence_length / self.num_sentences)
        print("self.single_sequence_length:",self.sequence_length,";self.total_sequence_length:",self.total_sequence_length,";self.num_sentences:",self.num_sentences)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        print("model====>:",self.model)
        if self.model=='gru':
            print("going to use model:gru model")
            self.logits_accusation,self.logits_article,self.logits_deathpenalty,self.logits_lifeimprisonment,self.logits_imprisonment = self.inference_gru()  # [None, self.label_size]. main computation graph is here.
        elif self.model=='c_gru':
            print("going to use model:c_gru model")
            self.logits_accusation, self.logits_article, self.logits_deathpenalty, self.logits_lifeimprisonment, self.logits_imprisonment =self.inference_c_gru()
        elif self.model=='c_gru2':
            print("going to use model:c_gru2 model")
            self.logits_accusation, self.logits_article, self.logits_deathpenalty, self.logits_lifeimprisonment, self.logits_imprisonment =self.inference_c_gru2()
        elif self.model=='text_cnn':
            print("going to use model:text cnn model")
            self.logits_accusation, self.logits_article, self.logits_deathpenalty, self.logits_lifeimprisonment, self.logits_imprisonment =self.inference_text_cnn()
        elif self.model=='han':
            print("going to use model:hierarcial attention network")
            self.logits_accusation,self.logits_article,self.logits_deathpenalty,self.logits_lifeimprisonment,self.logits_imprisonment = self.inference_han()  # [None, self.label_size]. main computation graph is here.
        elif self.model=='dp_cnn':
            print("going to use model:deep pyramid cnn model.")
            self.logits_accusation,self.logits_article,self.logits_deathpenalty,self.logits_lifeimprisonment,self.logits_imprisonment = self.inference_deep_pyramid_cnn()  # [None, self.label_size]. main computation graph is here.
        else:
            print("going to use model: pooling")
            self.logits_accusation, self.logits_article, self.logits_deathpenalty, self.logits_lifeimprisonment, self.logits_imprisonment = self.inference_pooling()
        #if not is_training:
        #    return
        #if is_training is not None:
        #    self.is_training = is_training
        #else:
        #    self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.loss_val = self.loss()
        self.train_op = self.train()

    def inference_han(self):
        """hierarcial attention network:main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.dropout 6.transform for each task 7.linear classifier"""
        # 1.Word Encoder: use bilstm to encoder the sentence.
        embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  # [None,num_sentences,sentence_length,embed_size]
        embedded_words=[tf.squeeze(x) for x in tf.split(embedded_words,self.num_sentences,axis=1)] #a list.length is num_sentence, each element is:[None,sentence_length,embed_size]
        word_attention_list=[]
        for i in range(self.num_sentences):
            sentence=embedded_words[i]
            #sentence is:[batch_size,seqence_length,embed_size]
            resue_flag=True if i>0 else False
            # 2. word encoder
            num_units1=self.embed_size
            sentence=tf.reshape(sentence,(-1,self.sequence_length,num_units1))
            word_encoded,word_encodeded2=self.bi_lstm(sentence, 'word_level', self.hidden_size,reuse_flag=resue_flag) #[batch_size,seq_length,num_units*2].
            # 3. word attention
            #word_encoded=tf.transpose(word_encoded,perm=[0,2,1]) #[batch_size,num_units*2]
            if i == 0: print("1.#############word_encoded:",word_encoded)
            word_attention=self.attention_multihop( word_encoded, 'word_level', reuse_flag=resue_flag)  #[batch_size,num_units*2]
            word_attention = self.layer_normalization(word_attention, i, 'word_level') #TODO add layer normalization 2018-05-31
            if i == 0: print("2.#############word_attention:",word_attention)
            word_attention = tf.nn.dropout(word_attention, keep_prob=self.dropout_keep_prob) #TODO add 2018-05-27
            #word_attention=tf.concat([word_attention,word_encodeded2],axis=1) #TODO add 2018-05-27
            word_attention_list.append(word_attention)

        sentence_encoder_input=tf.stack(word_attention_list,axis=1) #[batch_size,num_sentence,num_units*2]
        print("sentence_encoder_input:",sentence_encoder_input)
        # 4. sentence encoder
        print("3.#############.sentence_encoder_input:",sentence_encoder_input)
        sentence_encoded,sentence_encoded2 = self.bi_lstm(sentence_encoder_input, 'sentence_level',self.hidden_size*2)  # [batch_size,seq_length,num_units*4]
        print("4.#############.sentence_encoded:",sentence_encoded)
        # 5. sentence attention
        document_representation=self.attention_multihop(sentence_encoded,'sentence_level') #attention. [batch_size,num_units*4]. TODO add multi-head 2018-05-31

        document_representation=self.layer_normalization(document_representation, 2, 'sentence_level') #TODO add layer normalization 2018-05-31
        print("5.#############document_representation:",document_representation)
        # 6. dropout
        h = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)  # [batch_size,num_units*4]
        #h = tf.concat([h, sentence_encoded2], axis=1) #TODO add 2018-05-27
        print("6.#############h:",h)

        logits_list=self.project_tasks(h)
        return logits_list

    def inference_gru(self):
        """
        gru based model: embedding--->bi-gru(with LN)-->ui-gru(two layers)-->last hidden state(with LN)-->projection-->predict
        :return:
        """
        #0. embedding
        embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  # [None,num_sentences,sentence_length,embed_size]
        #1. bi-gru and get last hidden state
        encoded_sequences, last_hidden_state = self.bi_lstm(embedded_words, 'bi_lstm', self.hidden_size)  # [batch_size,seq_length,hidden_size*2],[batch_size, num_units]
        encoded_sequences=self.layer_normalization(encoded_sequences,1,'bi_lstm') # [batch_size,seq_length,hidden_size*2]
        #2. two layer of uni-gru
        gru_hidden_size=encoded_sequences.get_shape().as_list()[-1] #hidden_size*2
        encoded_sequences2,last_hidden_state2=self.un_lstm(encoded_sequences,'uni_lstm',gru_hidden_size,num_layers=3)
        print("last_hidden_stateTwo:",last_hidden_state2)
        #('last_hidden_state2:', (LSTMStateTuple(c=<tf.Tensor 'bi_lstm_uni_lstm/rnn/while/Exit_3:0' shape=(128, 128) dtype=float32>, #TODO why has two last hidden states.
                                     # h=<tf.Tensor 'bi_lstm_uni_lstm/rnn/while/Exit_4:0' shape=(128, 128) dtype=float32>),
        #                        LSTMStateTuple(c=<tf.Tensor 'bi_lstm_uni_lstm/rnn/while/Exit_5:0' shape=(128, 128) dtype=float32>,
        #                             h=<tf.Tensor 'bi_lstm_uni_lstm/rnn/while/Exit_6:0' shape=(128, 128) dtype=float32>))
        #last_hidden_state2=tf.concat([last_hidden_state2[0][1],last_hidden_state2[1][1]],axis=1) #[batch_size,hidden_size*2]
        last_hidden_state2=last_hidden_state2[-1][1] ##[batch_size,hidden_size*2]
        last_demension=last_hidden_state2.get_shape().as_list()[-1]
        last_hidden_state2=tf.reshape(last_hidden_state2,(-1,1,last_demension))
        last_hidden_state2=self.layer_normalization(last_hidden_state2,2,'uni_lstm')
        h=tf.squeeze(last_hidden_state2)
        #3. FC layers
        h=tf.layers.dense(h,self.hidden_size*4,activation=tf.nn.tanh,use_bias=True)
        #4.. classfier
        logits_list = self.project_tasks(h)
        return logits_list

    def inference_c_gru(self):#CNN-->RNN-->FC->Classifier
        """
        combined CNN and RNN: cnn followed by rnn(gru). CNN is used to reduce timestamp; RNN is used to learn long distance dependency.
        :return:
        """
        #1. cnn to reduce length of input: 400--->30
        input_x=tf.nn.embedding_lookup(self.Embedding,self.input_x)
        cnn_result= self.conv_layers_return_3d(input_x, 'conv_layer', reuse_flag=False) #[batch_size,sequence_length-filter_size + 1,num_filters]. num_filters=hidden_size
        #2. rnn to get long term depency
        sequences_states,last_hidden_state=self.bi_lstm(cnn_result, 'rnn_level', self.hidden_size * 2) #last hidden state:[batch_size, hidden_size*4]
        #4. classifier
        h=tf.layers.dense(last_hidden_state,self.hidden_size*4,activation=tf.nn.tanh,use_bias=True) #[batch_size, hidden_size*4]
        logits_list = self.project_tasks(h)
        return logits_list

    def inference_c_gru2(self):# RNN-->CNN-->FC-->Classifier
        """
        combined CNN and RNN: cnn followed by rnn(gru). CNN is used to reduce timestamp; RNN is used to learn long distance dependency.
        :return:
        """
        #1. rnn to get long term depency
        embedded_input = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        sequences_states,last_hidden_state=self.bi_lstm(embedded_input, 'rnn_level', self.hidden_size ) #sequences_states:[batch_size,seq_length,hidden_size*2]
        sequences_states=self.layer_normalization(sequences_states,1,'bi_lstm') # [batch_size,seq_length,hidden_size*2]

        #1. cnn to reduce length of input: 400--->30
        cnn_result= self.conv_layers_return_3d(sequences_states, 'conv_layer', reuse_flag=False) #[batch_size,sequence_length-filter_size + 1,num_filters]=[batch_size,sequence_length-filter_size + 1,hidden_size]
        # max pooling
        pooling=tf.nn.max_pool(cnn_result, ksize=[1,((self.total_sequence_length-self.filter_sizes[0])/self.stride_length)+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
        #4. classifier
        h=tf.layers.dense(pooling,self.hidden_size*4,activation=tf.nn.tanh,use_bias=True)
        logits_list = self.project_tasks(h)
        return logits_list

    def inference_text_cnn(self):# RNN-->CNN-->NN-->Classifier
        """
        combined CNN and RNN: cnn followed by rnn(gru). CNN is used to reduce timestamp; RNN is used to learn long distance dependency.
        :return:
        """
        input_x = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        #input_x = tf.layers.dense(input_x, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        cnn = self.conv_layers_return_2d_great(input_x, 'conv_layer',reuse_flag=False)  # [batch_size,sequence_length-filter_size + 1,num_filters]
        #pooling=tf.nn.max_pool(cnn_result, ksize=[1,((self.total_sequence_length-self.filter_size[0])/self.stride_length)+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
        # 4. classifier
        h = tf.layers.dense(cnn, self.hidden_size * 4, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        logits_list = self.project_tasks(h)
        return logits_list

    def inference_deep_pyramid_cnn(self):
        """
        deep pyramid cnn for text categorization
        region embedding-->two layers of convs-->repeat of building block(Pooling,/2-->conv-->conv)--->pooling
        for more check: http://www.aclweb.org/anthology/P/P17/P17-1052.pdf
        :return: logits_list
        """
        # 1.region embedding
        embedding_documents = self.region_embedding()  # shape:[batch_size,total_sequence_length,embedding_size]

        # 2.two layers of convs
        embedding_documents = tf.expand_dims(embedding_documents, -1)  # [batch_size,total_sequence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        conv = self.dpcnn_two_layers_conv(embedding_documents)  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        # 2.1 skip connection: add and activation
        conv = conv + embedding_documents  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        b = tf.get_variable("b-inference", [self.hpcnn_number_filters])
        conv = tf.nn.relu(tf.nn.bias_add(conv, b), "relu-inference")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]

        # 3.repeat of building blocks
        for i in range(self.num_repeat):
            conv = self.dpcnn_pooling_two_conv(conv, i)  # shape:[batch_size,total_sequence_length/np.power(2,i),hpcnn_number_filters]

        # 4.max pooling
        seq_length1 = conv.get_shape().as_list()[1]  # sequence length after multiple layers of conv and pooling
        seq_length2 = conv.get_shape().as_list()[2]  # sequence length after multiple layers of conv and pooling
        print("before.final.pooling:", conv) #(256, 25, 4, 16)
        pooling = tf.nn.max_pool(conv, ksize=[1, seq_length1, seq_length2, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool")  # [batch_size,hpcnn_number_filters]
        pooling = tf.squeeze(pooling) #(256, 16)
        print("pooling.final:", pooling)

        # 5.classifier
        h = tf.layers.dense(pooling, self.hidden_size * 4, activation=tf.nn.relu, use_bias=True) #[batch_size,h*4]
        h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob) #[batch_size,h*4]
        logits_list = self.project_tasks(h)
        return logits_list

        return pooling

    def inference_text_cnn_two_embedding(self):# RNN-->CNN-->NN-->Classifier
        """
        combined CNN and RNN: cnn followed by rnn(gru). CNN is used to reduce timestamp; RNN is used to learn long distance dependency.
        :return:x
        """
        input_x1 = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[batch_size,total_sequence_length,embed_size]
        input_x2 = tf.nn.embedding_lookup(self.Embedding2, self.input_x) #[batch_size,total_sequence_length,embed_size]
        input_x=tf.stack([input_x1,input_x2],axis=-1) #[batch_size,total_sequence_length,embed_size,2]
        #input_x = tf.layers.dense(input_x, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        cnn = self.conv_layers_return_2d_two_embedding(input_x, 'conv_layer',reuse_flag=False)  # [batch_size,sequence_length-filter_size + 1,num_filters]
        #pooling=tf.nn.max_pool(cnn_result, ksize=[1,((self.total_sequence_length-self.filter_size[0])/self.stride_length)+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
        # 4. classifier
        h = tf.layers.dense(cnn, self.hidden_size * 4, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        logits_list = self.project_tasks(h)
        return logits_list

    def inference_pooling(self):
        input_x = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[batch_size,sequence_length,embed_size]
        h=self.pooling(input_x,self.pooling_strategy)
        h = tf.layers.dense(h, self.hidden_size , activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        logits_list = self.project_tasks(h)
        return logits_list


    def dpcnn_pooling_two_conv(self, conv, layer_index):
        """
        pooling followed with two layers of conv, used by deep pyramid cnn.
        pooling-->conv-->conv-->skip connection
        conv:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        :return:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
        """
        with tf.variable_scope("pooling_two_conv_" + str(layer_index)):
            # 1. pooling:max-pooling with size 3 and stride 2==>reduce shape to half
            pooling = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name="pool")  # [batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
            print(layer_index, "dpcnn_pooling_two_conv.pooling:", pooling)

            # 2. two layer of conv
            conv = self.dpcnn_two_layers_conv(pooling,double_num_filters=False) #TODO double num_filters
            # print("dpcnn_pooling_two_conv.layer_index", layer_index, "conv:", conv)

            # 3. skip connection and activation
            conv = conv + pooling
            b = tf.get_variable("b-poolcnn%s" % self.hpcnn_number_filters, [self.hpcnn_number_filters])
            conv = tf.nn.relu(tf.nn.bias_add(conv, b),"relu-poolcnn")  # shape:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
        return conv

    def dpcnn_two_layers_conv(self, inputs,double_num_filters=False):
        """
        two layers of conv
        inputs:[batch_size,total_sequence_length,embed_size,dimension]. e.g.(128, 400, 64,1)-->[128,200,32,250]
        :return:[batch_size,total_sequence_length,embed_size,num_filters]
        """
        # conv1:
        # filter1's first three dimension apply to [total_sequence_length, embed_size, 1] of embedding_documents
        print("dpcnn_two_layers_conv.inputs:", inputs)  # (128, 400, 64, 250)
        channel = inputs.get_shape().as_list()[-1]
        if double_num_filters:
            hpcnn_number_filters =channel * 2
        else:
            hpcnn_number_filters=self.hpcnn_number_filters
        filter1 = tf.get_variable("filter1-%s" % self.hpcnn_filter_size,[self.hpcnn_filter_size, 1, channel, hpcnn_number_filters],initializer=self.initializer)
        conv1 = tf.nn.conv2d(inputs, filter1, strides=[1, self.stride_length, 1, 1], padding="SAME",name="conv")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        conv1 = tf.contrib.layers.batch_norm(conv1, is_training=self.is_training_flag, scope='cnn1')

        print("dpcnn_two_layers_conv.conv1:", conv1)  # (128, 400, 64, 250)
        b1 = tf.get_variable("b-cnn-%s" % hpcnn_number_filters, [hpcnn_number_filters])
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1),"relu1")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]

        # conv2
        # filter2's first three dimension apply to:[total_sequence_length,embed_size,hpcnn_number_filters] of conv1
        filter2 = tf.get_variable("filter2-%s" % self.hpcnn_filter_size,[self.hpcnn_filter_size, 1, hpcnn_number_filters, hpcnn_number_filters],initializer=self.initializer)
        conv2 = tf.nn.conv2d(conv1, filter2, strides=[1, self.stride_length, 1, 1], padding="SAME",name="conv2")  # shape:[batch_size,stotal_sequence_length,embed_size,hpcnn_number_filters]
        conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')

        print("dpcnn_two_layers_conv.conv2:", conv2)  # (128, 400, 64, 250)
        return conv2  # shape:[batch_size,total_sequence_length,embed_size,num_filters]

    def region_embedding(self):
        """
        region embedding for hp_cnn: embedding of a region of text covering one or more words.
        check: Enhancing region embedding with unsuper- vised embeddings in paper: deep pyramid cnn for text categorization
        instead of follow the way in the paper, here we just use pretrained word embedding
        :return:#[batch_size,sequence_length,embed_size]
        """
        embedded_document = tf.nn.embedding_lookup(self.Embedding,self.input_x)  # [batch_size,sequence_length,embed_size]
        return embedded_document  # [batch_size,sequence_length,embed_size]

    def pooling(self,inputs,pooling_name,use_mlp=True,n_gram=2):
        """
         :param inputs: [batch_size,sequence_length,embed_size]
        :param pooling_name:
        :param use_mlp: whether transform input using Multi-Layer Perception
        :return:
        """
        result=None
        if use_mlp:
            input=tf.layers.dense(inputs,self.hidden_size,activation=tf.nn.relu,use_bias=True) #[batch_size,sequence_length,hidden_size]
        if pooling_name=='avg':
            result=tf.reduce_mean(input,axis=1) #[batch_size,embed_size]
        elif pooling_name=='max':
            result=tf.reduce_max(input,axis=1) #[batch_size,embed_size]
        elif pooling_name=='concat':
            result_mean=tf.reduce_mean(input,axis=1) #[batch_size,embed_size]
            result_max=tf.reduce_max(input,axis=1) #[batch_size,embed_size]
            result=tf.concat([result_mean,result_max],axis=1) #[batch_size,embed_size*2]
        elif pooling_name=='hier':
            inputs_split=[tf.squeeze(x) for x in tf.split(inputs,self.total_sequence_length,axis=1)]
            context_list=[]
            for i, e in enumerate(inputs_split):
                if (i + (n_gram-1)) < self.total_sequence_length:#TODO bi-gram.
                    context = tf.stack([inputs_split[i], inputs_split[i + (n_gram-1)]], axis=1) #TODO bi-gram. [batch_size,2,hidden_size]
                    context_pooling=tf.squeeze(tf.reduce_mean(context,axis=1)) #[batch_size,hidden_size]<---[batch_size,1,hidden_size]
                    context_list.append(context_pooling) #context_pooling:[batch_size,hidden_size]
            representation=tf.stack(context_list,axis=1) #[batch_size,self.total_sequence_length-1,hidden_size]
            result=tf.reduce_max(representation,axis=1) #[batch_size,hidden_size]
        return result

    def un_lstm(self, input_sequences, level,num_units, num_layers=2,reuse_flag=False):
        """
        :param input_sequences: [batch_size,seq_lenght,num_units]
        :param level: word or sentence
        :param reuse_flag: resuse or not
        :return: encoded representation:[batch_size,seq_lenght,num_units*2]
        """
        with tf.variable_scope("bi_lstm_" + str(level), reuse=reuse_flag):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=0.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
            outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=input_sequences, dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        return outputs,hidden_states  #[batch_size,sequence_length,num_units*2]


    def inference_self_attention(self):
        #use attention is all you need: transformer. self-attention,multi-head attention,postion-wise layer,layer normalization, residual connection.
        pass

    def project_tasks(self,h):
        """
        :param h: shared features
        :return: logits
        transoform each sub task using one-layer MLP ,then get logits.
        get some insights from densely connected layers from recently development
        """
        #1.accusation: FC-->dropout-->classifier
        h_accusation = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h_accusation = tf.nn.dropout(h_accusation,keep_prob=self.dropout_keep_prob) # TODO ADD 2018.07.02
        logits_accusation = tf.layers.dense(h_accusation, self.accusation_num_classes,use_bias=True)  # shape:[None,self.num_classes]

        #2.relevant article: concated features-->FC-->dropout-->classifier
        h_article_concated=tf.concat([h,h_accusation],axis=-1) #TODO [batch,?,hidden_size*2] ADD 2018.07.02
        h_article = tf.layers.dense(h_article_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h_article = tf.nn.dropout(h_article,keep_prob=self.dropout_keep_prob) # TODO ADD 2018.07.02
        logits_article = tf.layers.dense(h_article, self.article_num_classes,use_bias=True)  # shape:[None,self.num_classes]

        #3.death penalty: concated features-->FC-->dropout-->classifier
        h_deathpenalty_concated=tf.concat([h,h_accusation,h_article],axis=-1)  #TODO [batch,?,hidden_size*3] ADD 2018.07.02
        h_deathpenalty = tf.layers.dense(h_deathpenalty_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h_deathpenalty = tf.nn.dropout(h_deathpenalty,keep_prob=self.dropout_keep_prob) # TODO ADD 2018.07.02
        logits_deathpenalty = tf.layers.dense(h_deathpenalty,self.deathpenalty_num_classes,use_bias=True)  # shape:[None,self.num_classes] #

        #4.life imprisonment: concated features-->FC-->dropout-->classifier
        h_lifeimprsion_concated=tf.concat([h,h_accusation,h_article,h_deathpenalty],axis=-1)
        h_lifeimprisonment = tf.layers.dense(h_lifeimprsion_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h_lifeimprisonment = tf.nn.dropout(h_lifeimprisonment,keep_prob=self.dropout_keep_prob) # TODO ADD 2018.07.02
        logits_lifeimprisonment = tf.layers.dense(h_lifeimprisonment, self.lifeimprisonment_num_classes,use_bias=True)  # shape:[None,self.num_classes]

        #5.imprisonment: concated features-->FC-->dropout-->classifier
        h_imprisonment_concated=tf.concat([h,h_accusation,h_article,h_deathpenalty,h_lifeimprisonment],axis=-1)
        logits_imprisonment = tf.layers.dense(h_imprisonment_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        logits_imprisonment = tf.nn.dropout(logits_imprisonment,keep_prob=self.dropout_keep_prob) # TODO ADD 2018.07.02
        logits_imprisonment = tf.layers.dense(logits_imprisonment, 1,use_bias=True)  # imprisonment is a continuous value, no need to use activation function
        logits_imprisonment = tf.reshape(logits_imprisonment, [-1]) #[batch_size]
        return logits_accusation, logits_article, logits_deathpenalty, logits_lifeimprisonment, logits_imprisonment

    def conv_layers_return_3d(self, input_x, name_scope, reuse_flag=False):
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                print("conv:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                pooled_outputs.append(h) #h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        h=tf.squeeze(pooled_outputs[0]) #TODO temp only has one filter map, use first one [batch_size,sequence_length - filter_size + 1,num_filters]
        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h # [batch_size,sequence_length - filter_size + 1,num_filters]

    def conv_layers_return_2d_great(self, input_x, name_scope, reuse_flag=False):#great 81.3
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                #1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv=tf.contrib.layers.batch_norm(conv,is_training=self.is_training_flag,scope='cnn1')
                print(i,"conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                #2) CNN->BN->relu
                h=tf.reshape(h,[-1,self.total_sequence_length-filter_size+1,self.num_filters,1]) #shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                #Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2=tf.nn.conv2d(h,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") #shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2=tf.contrib.layers.batch_norm(conv2,is_training=self.is_training_flag, scope='cnn2')
                print(i,"conv2:",conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                #3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, (self.total_sequence_length - filter_size*2+2), 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                #pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i,"pooling:",pooling_max)
                #pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max) #h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        #concat
        h=tf.concat(pooled_outputs,axis=1) #[batch_size,num_total_filters]
        print("h.concat:",h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h # [batch_size,sequence_length - filter_size + 1,num_filters]

    def conv_layers_return_2d_two_embedding(self, input_x, name_scope, reuse_flag=False):  # great 81.3

        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier
        input_x:[batch_size,sequence_length,embed_size,2]
        """
        # 1.=====>get emebedding of words in the sentence
        #sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        print("going to start:conv_layers_return_2d_two_embedding. input_x:",input_x)
        sentence_embeddings_expanded=input_x
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 2, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.total_sequence_length - filter_size + 1, self.num_filters, 1])  # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="VALID",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, (self.total_sequence_length - filter_size * 2 + 2), 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_total_filters]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]


    def conv_layers_return_2d(self, input_x, name_scope, reuse_flag=False):#great 81.3
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                #1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv=tf.contrib.layers.batch_norm(conv,is_training=self.is_training_flag,scope='cnn1')
                print(i,"conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]

                #2) CNN->BN->relu
                h=tf.reshape(h,[-1,self.total_sequence_length-filter_size+1,self.num_filters,1]) #shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                #Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2=tf.nn.conv2d(h,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") #shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters*2]
                conv2=tf.contrib.layers.batch_norm(conv2,is_training=self.is_training_flag, scope='cnn2')
                print(i,"conv2:",conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size*2 + 2,1,num_filters*2]. tf.nn.bias_add:adds `bias` to `value`

                #3) CNN->BN->relu
                #h=tf.reshape(h,[-1,self.total_sequence_length-filter_size*2+2,self.num_filters,1]) #shape:[batch_size,sequence_length-filter_size*2+2,num_filters,1]
                #Layer2:CONV-RELU
                #filter3 = tf.get_variable("filter3-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                #conv3=tf.nn.conv2d(h,filter3,strides=[1,1,1,1],padding="VALID",name="conv3") #shape:[batch_size,sequence_length-filter_size*3+3,1,num_filters]
                #conv3=tf.contrib.layers.batch_norm(conv3,is_training=self.is_training_flag, scope='cnn3') #shape:[batch_size,sequence_length-filter_size*3+3,1,num_filters]
                #print(i,"conv3:",conv3) #(128, 385, 1, 128)
                #b3 = tf.get_variable("b3-%s" % filter_size, [self.num_filters])  #shape:[batch_size,sequence_length-filter_size*3+3,1,num_filters]

                #conv3=conv3+conv2[:,0:(self.total_sequence_length - filter_size*3+3),:,:] #TODO TEST IT, OTHERWISE NEED TO REMOVE ...........................
                #h = tf.nn.relu(tf.nn.bias_add(conv3, b3),"relu3")  #shape:[batch_size,sequence_length-filter_size*3+3,1,num_filters]

                #3). Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, (self.total_sequence_length - filter_size*2+2), 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                #pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i,"pooling:",pooling_max)
                #pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max) #h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        #concat
        h=tf.concat(pooled_outputs,axis=1) #[batch_size,num_total_filters]
        print("h.concat:",h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h # [batch_size,sequence_length - filter_size + 1,num_filters]


    def conv_layers_return_2d_3layer(self, input_x, name_scope, reuse_flag=False):#CNN(BN->RELU)-->CNN(BN->RELU)-->Pooling-->CNN(BN-->RELU)-->Pooling
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                print(i,"conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.total_sequence_length - filter_size + 1, self.num_filters,1])  # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="VALID",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i,"conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3) Max-pooling, 2
                h = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1") #[batch_size,(sequence_length - filter_size + 1)/2,num_filters]
                print(i, "pooling1:", h) #(128, 389, 1, 128)

                # 4) CNN->BN->relu
                length_pooled=h.get_shape().as_list()[1]
                h = tf.reshape(h, [-1, length_pooled, self.num_filters,1])  # shape:[batch_size,(sequence_length-filter_size+1)/2,num_filters,1]
                # Layer2:CONV-RELU
                filter3 = tf.get_variable("filter3-%s" % filter_size,[filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv3 = tf.nn.conv2d(h, filter3, strides=[1, 1, 1, 1], padding="VALID",name="conv3")
                conv3 = tf.contrib.layers.batch_norm(conv3, is_training=self.is_training_flag, scope='cnn3') ## shape:[batch_size,((sequence_length-filter_size+1)/2)-num_filters+1,num_filters,1]
                print(i,"conv3:", conv3)
                b3 = tf.get_variable("b3-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv3, b3),"relu3")  #

                # 5) Max-pooling
                length_pooled2=h.get_shape().as_list()[1] #[batch_size, (sequence_length-filter_size+1)/2)-filter_size+1,1,num_filters]
                h = tf.nn.max_pool(h, ksize=[1, length_pooled2, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool2") #[batch_size,(sequence_length - filter_size + 1)/2,num_filters]
                print(i,"h:",h)
                h=tf.squeeze(h)
                pooled_outputs.append(h)  # h:[batch_size,sequence_length - filter_size + 1,1,num_filters]

        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_total_filters]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h,keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]


    def conv_layers_leNet5(self, input_x, name_scope, reuse_flag=False):
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                #1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                #conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)
                cnn=tf.contrib.layers.batch_norm(conv,is_training=self.is_training_flag,scope='cnn1')
                print("conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                #2) CNN->BN->relu
                h=tf.reshape(h,[-1,self.total_sequence_length-filter_size+1,self.num_filters,1]) #shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                #Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2=tf.nn.conv2d(h,filter2,strides=[1,1,1,1],padding="VALID",name="conv2") #shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                cnn2=tf.contrib.layers.batch_norm(conv2,is_training=self.is_training_flag, scope='cnn2')

                print("conv2:",conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                #3. Max-pooling
                pooling = tf.nn.max_pool(h, ksize=[1, (self.total_sequence_length - filter_size*2+2), 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooling=tf.squeeze(pooling) #[batch_size,seq_length',num_filters]
                print(i,"pooling:",pooling)
                pooled_outputs.append(pooling) #h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        #concat
        h=tf.concat(pooled_outputs,axis=1) #[batch_size,num_total_filters]
        print("h.concat:",h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h # [batch_size,sequence_length - filter_size + 1,num_filters]


    def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=False): #check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py#L89
        """
        batch normalization: keep moving average of mean and variance. use it as value for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def attention(self,input_sequences,attention_level,reuse_flag=False):
        """
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        num_units=input_sequences.get_shape().as_list()[-1] #get last dimension
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units],initializer=self.initializer)
            #1.one-layer MLP
            u=tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True) #[batch_size,seq_legnth,num_units].no-linear
            #2.compute weight by compute simility of u and attention vector v
            score=tf.multiply(u,v_attention) #[batch_size,seq_length,num_units]
            weight=tf.reduce_sum(score,axis=2,keepdims=True) #[batch_size,seq_length,1]
            #weight=tf.nn.softmax(weight,axis=1) #[batch_size,seq_length,1] #TODO temp removed since it make performance worse 2018.05.29
            #3.weight sum
            attention_representation=tf.reduce_sum(tf.multiply(u,weight),axis=1) #[batch_size,num_units]. TODO here we not use original input_sequences but transformed version of input: u.
        return attention_representation

    def attention_multihop(self,input_sequences,attention_level,reuse_flag=False):
        """
        perform multi-hop attention, instead of only one hop. but final demsion is same as before.
        :param input_sequences:[batch_size,sequence_length,num_units]
        :param attention_level:
        :param reuse_flag:
        :return: attention_representation:[batch_size,sequence_length,num_units]
        """
        num_hops=4
        num_units = input_sequences.get_shape().as_list()[-1]/num_hops  # get last dimension
        attention_rep_list=[]
        for i in range(num_hops):
            with tf.variable_scope("attention_"+str(i) +'_'+ str(attention_level), reuse=reuse_flag):
                v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units], initializer=self.initializer)
                # 1.one-layer MLP
                u = tf.layers.dense(input_sequences, num_units, activation=tf.nn.tanh,use_bias=True)  # [batch_size,seq_legnth,num_units].no-linear
                # 2.compute weight by compute simility of u and attention vector v
                score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
                weight = tf.reduce_sum(score, axis=2, keepdims=True)  # [batch_size,seq_length,1]
                # weight=tf.nn.softmax(weight,axis=1) #[batch_size,seq_length,1] #TODO temp removed since it make performance worse 2018.05.29
                # 3.weight sum
                attention_rep = tf.reduce_sum(tf.multiply(u, weight),axis=1)  # [batch_size,num_units]. TODO here we not use original input_sequences but transformed version of input: u.
                attention_rep_list.append(attention_rep)

        attention_representation=tf.concat(attention_rep_list,axis=-1) #[
        return attention_representation

    def layer_normalization(self,x,layer_index,type):
        """
        x should be:[batch_size,sequence_length,d_model]
        :return:
        """
        filter=x.get_shape()[-1] #last dimension of x. e.g. 512
        with tf.variable_scope("layer_normalization"+type+str(layer_index)):
            # 1. normalize input by using  mean and variance according to last dimension
            mean=tf.reduce_mean(x,axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            variance=tf.reduce_mean(tf.square(x-mean),axis=-1,keep_dims=True) #[batch_size,sequence_length,1]
            norm_x=(x-mean)*tf.rsqrt(variance+1e-6) #[batch_size,sequence_length,d_model]
            # 2. re-scale normalized input back
            scale=tf.get_variable("layer_norm_scale",[filter],initializer=tf.ones_initializer) #[filter]
            bias=tf.get_variable("layer_norm_bias",[filter],initializer=tf.ones_initializer) #[filter]
            output=norm_x*scale+bias #[batch_size,sequence_length,d_model]
            return output #[batch_size,sequence_length,d_model]

    def attention_multiply(self,input_sequences,attention_level,reuse_flag=False): #TODO need update
        """
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        num_units=input_sequences.get_shape().as_list()[-1] #get last dimension
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units],initializer=self.initializer)
            #1.one-layer MLP
            u=tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True) #[batch_size,seq_legnth,num_units].no-linear
            #2.compute weight by compute simility of u and attention vector v
            score=tf.multiply(u,v_attention) #[batch_size,seq_length,num_units]. TODO NEED ADD multiply SCALE V_a.
            score=tf.reduce_sum(score,axis=2,keepdims=True) #/tf.sqrt(tf.cast(num_units,tf.float32)) #[batch_size,seq_length,1]
            weight=tf.nn.softmax(score,axis=1) #[batch_size,seq_length,1]
            #3.weight sum
            attention_representation=tf.reduce_sum(tf.multiply(input_sequences,weight),axis=1) #[batch_size,num_units]
        return attention_representation

    def attention_additive_batch(self,input_sequences_original, attention_level,reuse_flag=False):  #TODO check: paper 'Neural Machine Transation By Jointly Learning To Align and Translate'

        """ additive attention(support batch of input with sequences)
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: #[batch_size,sequence_length]
        """
        # [batch_size,seq_length,num_units*2].
        input_sequences=tf.transpose(input_sequences_original,perm=[0,2,1]) #[batch_size,num_units,sequence_length]<---[batch_size,seq_length,num_units].
        _, num_units, sequence_lenghth = input_sequences.get_shape().as_list()
        print("###attention_additive_batch.input_sequences:",input_sequences,";attention_level:",attention_level,"num_units:", num_units, ";sequence_lenghth:", sequence_lenghth)
        with tf.variable_scope("attention_" + str(attention_level), reuse=reuse_flag):
            # 1.create or get learnable variables
            attention_vector = tf.get_variable("attention_vector_" + attention_level,shape=[num_units, 1],initializer=self.initializer)
            W = tf.get_variable("W" + attention_level,shape=[1, num_units, num_units],initializer=self.initializer)
            U = tf.get_variable("U" + attention_level, shape=[num_units, num_units],initializer=self.initializer)
            v = tf.get_variable("v" + attention_level, shape=[1, 1, num_units],initializer=self.initializer)

            # 2.get part1 and part2 of additive attention
            W = tf.tile(W, (self.batch_size, 1, 1))  # [batch_size,num_units,num_units]
            part1 = tf.matmul(W,input_sequences)  # [batch_size,num_units,sequence_length]<----([batch_size,num_units,num_units],[batch_size,num_units,sequence_length])
            part2 = tf.expand_dims(tf.matmul(U, attention_vector),axis=0)  # [1,num_units,1]<---[num_units,1]<-----([num_units,num_units],[num_units,1])

            # 3.activation
            activation = tf.nn.tanh(part1 + part2)  # [batch_size,num_units,sequence_length]

            # 4. get attention score by using matmul
            v = tf.tile(v, (self.batch_size, 1, 1))  # [batch_size,1,num_units]
            score = tf.matmul(v,activation)  # [batch_size,1,sequence_length]<------([batch_size,1,num_units],[batch_size,num_units,sequence_length])
            score = tf.squeeze(score)  # [batch_size,sequence_length]

            # 5. normalize using softmax
            weights=tf.nn.softmax(score,axis=1) #[batch_size,sequence_length]

            # 6. weighted sum
            weights=tf.expand_dims(weights,axis=-1) #[batch_size,sequence_length,1]
            result=tf.multiply(input_sequences_original,weights) #[batch_size,squence_length,num_units]
            result=tf.reduce_sum(result,axis=1) #[batch_size,num_units]
        return result  # [batch_size,num_units]

    def attention_additive(self,input_sequence,attention_level,reuse_flag=False): #check: paper 'Neural Machine Transation By Jointly Learning To Align and Translate'

        """
        :param input_sequence: [num_units,1]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        attention_representation=None

        num_units=input_sequence.get_shape().as_list()[-1] #get last dimension
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            attention_vector = tf.get_variable("attention_vector_" + attention_level, shape=[num_units,1],initializer=self.initializer)
            W=tf.get_variable("W" + attention_level, shape=[num_units,num_units],initializer=self.initializer)
            U=tf.get_variable("U" + attention_level, shape=[num_units,num_units],initializer=self.initializer)

            v = tf.get_variable("v" + attention_level, shape=[1,num_units],initializer=self.initializer)
            part1=tf.matmul(W,input_sequence)   #[num_units,1]<----([num_units,num_units],[num_units,1])
            part2=tf.matmul(U,attention_vector) #[num_units,1]<-----([num_units,num_units],[num_units,1])
            activation=tf.nn.tanh(part1+part2)  #[num_units,1]
            result=tf.matmul(v,activation) #  [1,1]<------([1,num_units],[num_units,1])
            result=tf.reshape(result,()) #scalar
        return result


    def bi_lstm(self, input_sequences, level,num_units, reuse_flag=False):
        """
        :param input_sequences: [batch_size,seq_lenght,num_units]
        :param level: word or sentence
        :param reuse_flag: resuse or not
        :return: encoded representation:[batch_size,seq_lenght,num_units*2]
        """
        #num_units=input_sequences.get_shape().as_list()[-1] #get last dimension
        batch_size=input_sequences.get_shape().as_list()[0]
        with tf.variable_scope("bi_lstm_" + str(level), reuse=reuse_flag):
            lstm_fw_cell = rnn.GRUCell(num_units)  # forward direction cell.BasicLSTMCell
            lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.GRUCell(num_units)  # backward direction cell
            lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
            initial_state_fw= tf.get_variable("initial_state_fw", shape=[batch_size,num_units],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            initial_state_bw= tf.get_variable("initial_state_bw", shape=[batch_size,num_units],initializer=self.initializer)
            outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_sequences,initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw,dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        #concat output
        encdoded_inputs = tf.concat(outputs, axis=2)  #[batch_size,sequence_length,hidden_size*2]
        encodeded_inputs2=tf.concat([hidden_states[0],hidden_states[1]],axis=1)
        return encdoded_inputs,encodeded_inputs2  #[batch_size,sequence_length,num_units*2]

    def loss(self,l2_lambda=0.0001*3,epislon=0.000001):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

        #loss1: accusation
        #input_y_accusation_onehot=tf.one_hot(self.input_y_accusation,self.accusation_num_classes)
        losses_accusation= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_accusation,logits=self.logits_accusation)  #[batch_size,num_classes]
        self.loss_accusation = tf.reduce_mean((tf.reduce_sum(losses_accusation,axis=1)))  # shape=(?,)-->(). loss for all data in the batch-->single loss

        #loss2:relevant article
        #input_y_article_onehot=tf.one_hot(self.input_y_article,self.article_num_classes)
        losses_article= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_article,logits=self.logits_article)  # [batch_size,num_classes]
        self.loss_article = tf.reduce_mean((tf.reduce_sum(losses_article, axis=1)))  # shape=(?,)-->(). loss for all data in the batch-->single loss

        #loss3:death penalty
        print("self.input_y_deathpenalty:",self.input_y_deathpenalty,";self.logits_deathpenalty:",self.logits_deathpenalty)
        losses_deathpenalty = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_deathpenalty,logits=self.logits_deathpenalty)
        self.loss_deathpenalty = tf.reduce_mean((tf.reduce_sum(losses_deathpenalty, axis=1))) # shape=(?,)-->(). loss for all data in the batch-->single loss

        #loss4:life imprisonment
        losses_lifeimprisonment = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_lifeimprisonment,logits=self.logits_lifeimprisonment)
        self.loss_lifeimprisonment = tf.reduce_mean((tf.reduce_sum(losses_lifeimprisonment, axis=1))) # shape=(?,)-->(). loss for all data in the batch-->single loss

        #loss5: imprisonment: how many year in prison.
        self.loss_imprisonment =tf.reduce_mean(tf.divide(tf.pow((self.logits_imprisonment-self.input_y_imprisonment),2),1000.0)) #1000.0TODO
        print("sigmoid_cross_entropy_with_logits.losses:", losses_accusation)  # shape=(?, 1999).
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        self.weights_accusation = 0.25 #tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 3.0    # 0--1/3
        self.weights_article = 0.25 #tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 3.0       # 0--1/3
        self.weights_deathpenalty = 0.15 #tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 9.0   #0--1/9
        self.weights_lifeimprisonment = 0.15 #tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 9.0 #0--1/9
        self.weights_imprisonment=0.2 #1-(self.weights_accusation+self.weights_article+self.weights_deathpenalty+self.weights_lifeimprisonment) #0-1/9
        loss = self.weights_accusation*self.loss_accusation+self.weights_article*self.loss_article+self.weights_deathpenalty*self.loss_deathpenalty +\
               self.weights_lifeimprisonment*self.loss_lifeimprisonment+self.weights_imprisonment*self.loss_imprisonment+self.l2_loss
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        #train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)

        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding2 = tf.get_variable("Embedding2", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)


batch_size=128
def attention_additive_batch(input_sequences, attention_level,reuse_flag=False):  #TODO check: paper 'Neural Machine Transation By Jointly Learning To Align and Translate'

    """ additive attention(support batch of input with sequences)
    :param input_sequence: [batch_size,seq_length,num_units]
    :param attention_level: word or sentence level
    :return: #[batch_size,sequence_length]
    """
    # [batch_size,seq_length,num_units*2].
    input_sequences=tf.transpose(input_sequences,perm=[0,2,1]) #[batch_size,num_units,sequence_length]<---[batch_size,seq_length,num_units].
    _, num_units, sequence_lenghth = input_sequences.get_shape().as_list()
    print("###attention_additive_batch.input_sequences:",input_sequences,";attention_level:",attention_level,"num_units:", num_units, ";sequence_lenghth:", sequence_lenghth)
    with tf.variable_scope("attention_" + str(attention_level), reuse=reuse_flag):
        # 1.create or get learnable variables
        attention_vector = tf.get_variable("attention_vector_" + attention_level,shape=[num_units, 1])#,initializer=self.initializer)
        W = tf.get_variable("W" + attention_level,shape=[1, num_units, num_units])#,initializer=self.initializer)
        U = tf.get_variable("U" + attention_level, shape=[num_units, num_units])#,initializer=self.initializer)
        v = tf.get_variable("v" + attention_level, shape=[1, 1, num_units])#,initializer=self.initializer)

        # 2.get part1 and part2 of additive attention
        W = tf.tile(W, (batch_size, 1, 1))  # [batch_size,num_units,num_units]
        part1 = tf.matmul(W,input_sequences)  # [batch_size,num_units,sequence_length]<----([batch_size,num_units,num_units],[batch_size,num_units,sequence_length])
        part2 = tf.expand_dims(tf.matmul(U, attention_vector),axis=0)  # [1,num_units,1]<---[num_units,1]<-----([num_units,num_units],[num_units,1])

        # 3.activation
        activation = tf.nn.tanh(part1 + part2)  # [batch_size,num_units,sequence_length]

        # 4. get attention score by using matmul
        v = tf.tile(v, (batch_size, 1, 1))  # [batch_size,1,num_units]
        score = tf.matmul(v,activation)  # [batch_size,1,sequence_length]<------([batch_size,1,num_units],[batch_size,num_units,sequence_length])
        score = tf.squeeze(score)  # [batch_size,sequence_length]

        # 5. normalize using softmax
        weights=tf.nn.softmax(score,axis=1) #[batch_size,sequence_length]

        # 6. weighted sum
        weights=tf.expand_dims(weights,axis=-1) #[batch_size,sequence_length,1]
        attention_vector=tf.reshape(attention_vector,(1,1,num_units)) #[1,1,num_units]
        result=tf.multiply(attention_vector,weights) #[batch_size,squence_length,num_units]
        result=tf.reduce_sum(result,axis=1) #[batch_size,num_units]
    return result  # [batch_size,num_units]

#input_sequences=tf.ones((128,40,64))
#result=attention_additive_batch(input_sequences, 'word')
#print("result:",result) #result should be:[128,64]