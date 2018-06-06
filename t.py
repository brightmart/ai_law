
def highway(input_, size, num_layers=1, bias=-2.0, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            # g = f(tf.linear(input_, size, scope='highway_lin_%d' % idx))
            with tf.variable_scope('highway_lin_%d' % idx):
                g = tf.layers.dense(input_, size, activation=tf.nn.relu,bias=bias)
                #t = tf.nn.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
                t=tf.layers.dense(input_, size, activation=tf.nn.sigmoid, use_bias=True)
            output = t * g + (1. - t) * input_
            input_ = output

return output







import tensorflow as tf
logits=tf.ones((128,200))
labels=tf.ones((128,200))
weights=tf.ones((128))


#binary clssificaiton: binary-->cross entropy
y_pred =tf.nn.sigmoid(logits)  # [batch_size, num_classes]
losses =-1.0 *(tf.reduce_sum(labels * tf.log(y_pred)+(1 -labels) * tf.log(1 -y_pred), axis=1)) #[batch_size]
losses = tf.multiply(losses,weights)  #[batch_size]
loss = tf.reduce_mean(losses) #[]

sess=tf.Session()
sess.run(loss)