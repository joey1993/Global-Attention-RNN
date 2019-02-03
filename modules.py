import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import sys


def LR_layer(LR_batch_ph,
             keep_prob_ph,
             target_ph,
             vocabulary_size,
             LAMBDA1,
             LAMBDA2,
             att_w):

    with tf.name_scope('LR_layer'):

        LR_batch_ph = tf.nn.dropout(LR_batch_ph, keep_prob_ph)
        ATT_W = att_w
        b = tf.Variable(tf.constant(0., shape=[1]))

        LR_y_hat = tf.nn.xw_plus_b(LR_batch_ph, ATT_W, b)
        LR_y_hat = tf.squeeze(LR_y_hat)
        lr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=LR_y_hat, labels=target_ph))
        regularizer = tf.nn.l2_loss(ATT_W)
        LR_loss = lr_loss
        #LR_loss = tf.reduce_mean(lr_loss + LAMBDA1 * regularizer)
        LR_loss = LAMBDA2 * LR_loss
        return ATT_W, LR_loss


def RNN_layer(HIDDEN_SIZE,
              batch_embedded,
              seq_len_ph):

    rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    tf.summary.histogram('RNN_outputs', rnn_outputs)

    if isinstance(rnn_outputs, tuple):
        rnn_outputs = tf.concat(rnn_outputs, 2)
    #rnn_outputs = tf.layers.batch_normalization(rnn_outputs)

    return rnn_outputs
    

def feedforward(inputs, 
                num_units=[512, 2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.
        
    Returns:
        A 3d tensor with the same shape and dtype as inputs
    '''
    inputs = tf.reshape(inputs, [-1, 1, num_units[0]])
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[2], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        #outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)

        outputs = tf.squeeze(outputs,1)
    
    return outputs

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
          `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.
      
    Returns:
        A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs
