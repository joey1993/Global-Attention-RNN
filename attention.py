import tensorflow as tf
import sys
from modules import feedforward

def attention(inputs, 
              attention_size):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
        attention_size: Linear size of the Attention weights.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        alphas: local attention scores
            `[batch_size, sequence_length]

    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas


def global_attention(inputs, 
                     batch, 
                     attention_size, 
                     vocaubulary_size, 
                     ATT_W, 
                     load_LR_model=False):

    """
    Global Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    
    Args:
        inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
        batch: The RNN inputs, batch of dataset.
        attention_size: Linear size of the Attention weights.
        vocaubulary_size: the vocaubulary size of train + test dataset
        ATT_W: The global attention weights.
                in case of joint trained logistic regression model, input its parameters to initialize global attention weights.
        load_LR_model: If true, global attention weights are initialized by LR model weights.

    Returns:
        The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        Betas: Global attention scores.
            `[vocabulary_size, 1]
    """

    if isinstance(inputs, tuple):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)
    hidden_size = inputs.shape[2].value
    inputs = tf.layers.batch_normalization(inputs)

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    if load_LR_model:
        betas = tf.squeeze(ATT_W)
        u = tf.nn.embedding_lookup(tf.abs(betas), batch)
    else:
        betas = tf.Variable(tf.random_normal([vocaubulary_size], stddev=0.1))
        u = tf.nn.embedding_lookup(betas, batch)

    alphas = tf.nn.softmax(u, name='alphas') 

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    output = feedforward(output, num_units=[hidden_size, 2*hidden_size, hidden_size])
    output = tf.layers.batch_normalization(output)

    return output, betas

def run_attention(rnn_outputs, 
                  ATTENTION_SIZE, 
                  batch_ph, 
                  GLOBAL_ATTENTION_SIZE, 
                  vocabulary_size, ATT_W, 
                  load_LR_model):

    with tf.name_scope('Attention_layer'):
        attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE)
        tf.summary.histogram('alphas', alphas)

    with tf.name_scope('Global_attention_layer'):
        global_attention_output, betas = global_attention(rnn_outputs, batch_ph, GLOBAL_ATTENTION_SIZE, vocabulary_size, ATT_W, load_LR_model) 

    return attention_output, alphas, global_attention_output, betas


def attention_selection(rnn_outputs, 
                        attention_output, 
                        global_attention_output, 
                        attention_choice, 
                        attention_size):

    """
    Global Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    
    Args:
        rnn_outputs: The Attention inputs.
                    Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
        attention_output: local attention outputs
        global_attention_output: global attention outputs.
        attention_choice: decide which attention layer to user.
        attention_size: In case local and global attentions are both selected, employ another attention layer to do feature 
        selection from these two outputs. 

    Returns:
        The Attention output `Tensor`.
    """


    if attention_choice == 0: #no attention layer added
        return tf.reduce_sum(rnn_outputs, 1)

    elif attention_choice == 1: #only local attention is added
        return attention_output

    elif attention_choice == 2:
        return global_attention_output

    elif attention_choice == 3:
        hidden_size = attention_output.shape[1].value
        #print(attention_output)
        #print(global_attention_output)
        attention_output = tf.reshape(attention_output, [-1, 1, hidden_size])
        global_attention_output = tf.reshape(global_attention_output, [-1, 1, hidden_size])
        inputs = tf.concat([attention_output, global_attention_output], 1) #(?, 2, hidden_size)
        

        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))


        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        #print(alphas)

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        #print(output)
        #sys.exit()
        return output

    else:
        return attention_output












