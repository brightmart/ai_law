import tensorflow as tf

class DP_CNN:

    def __init__(self,batch_size=128,total_sequence_length=400):
        self.hpcnn_filter_size=3
        self.hpcnn_number_filters=64
        self.stride_length=1
        self.vocab_size=10000
        self.embed_size=64
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        self.num_repeat=4
        self.is_training_flag=True
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

        self.input_x = tf.placeholder(tf.int32, [batch_size, total_sequence_length], name="input_x")
        #embedding_documents = tf.nn.embedding_lookup(self.Embedding,input_x)  # [None,num_sentences,sentence_length,embed_size]
        #self.dpcnn_two_layers_conv(embedding_documents)
        result=self.inference_deep_pyramid_cnn()
        print("result:",result)

    def inference_deep_pyramid_cnn(self):
        """
        deep pyramid cnn for text categorization
        region embedding-->two layers of convs-->repeat of building block(Pooling,/2-->conv-->conv)--->pooling
        for more check: http://www.aclweb.org/anthology/P/P17/P17-1052.pdf
        :return: logits_list
        """
        #1.region embedding
        embedding_documents=self.region_embedding() #shape:[batch_size,total_sequence_length,embedding_size]

        #2.two layers of convs
        embedding_documents = tf.expand_dims(embedding_documents ,-1)  # [batch_size,total_sequence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        conv=self.dpcnn_two_layers_conv(embedding_documents,double_num_filters=False) #shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        #skip connection: add and activation
        conv=conv+embedding_documents #shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
        b = tf.get_variable("b-inference", [self.hpcnn_number_filters])
        print("conv:",conv,";b:",b)
        conv = tf.nn.relu(tf.nn.bias_add(conv, b),"relu-inference") #shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]

        #3. repeat of building blocks
        for i in range(self.num_repeat):
            conv=self.dpcnn_pooling_two_conv(conv,i) #shape:[batch_size,total_sequence_length/np.power(2,i),hpcnn_number_filters]

        #4. max pooling
        seq_length1=conv.get_shape().as_list()[1] #sequence length after multiple layers of conv and pooling
        seq_length2=conv.get_shape().as_list()[2] #sequence length after multiple layers of conv and pooling
        print("before.final.pooling:",conv)
        pooling=tf.nn.max_pool(conv, ksize=[1,seq_length1,seq_length2,1], strides=[1,1,1,1], padding='VALID',name="pool")   #[batch_size,hpcnn_number_filters]
        pooling=tf.squeeze(pooling)
        print("pooling.final:",pooling)

        #5. classifier

        return pooling

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
            print(layer_index,"dpcnn_pooling_two_conv.pooling:", pooling)

            # 2. two layer of conv
            conv = self.dpcnn_two_layers_conv(pooling)
            #print("dpcnn_pooling_two_conv.layer_index", layer_index, "conv:", conv)

            # 3. skip connection and activation
            conv = conv + pooling
            b = tf.get_variable("b-poolcnn%s" % self.hpcnn_number_filters, [self.hpcnn_number_filters])
            conv = tf.nn.relu(tf.nn.bias_add(conv, b),"relu-poolcnn")  # shape:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
        return conv

    def dpcnn_two_layers_conv(self, inputs,double_num_filters=True):
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
        embedded_document = tf.nn.embedding_lookup(self.Embedding, self.input_x) #[batch_size,sequence_length,embed_size]
        return embedded_document #[batch_size,sequence_length,embed_size]


x=DP_CNN()
print(x)