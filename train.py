#/usr/bin/python
"""
Attention RNN models

Train RNN (GRU) on ADVERTISER-specific user activity trail dataset (binary classification)
"""
from __future__ import print_function, division

import numpy as np
import sys
import tensorflow as tf

from tqdm import tqdm
from sklearn import metrics

from attention import attention,attention_selection,global_attention,run_attention
from modules import LR_layer,RNN_layer
from utils import *



tf.set_random_seed(2019)
np.random.seed(2019)

load_pretrained_embedding = False
attention_choice = 2 #[0: RNN only; 1: local attention only; 2: global attention only; 3: local + global attentions;]
load_LR_model = False
load_LR_initializer = False


# model hyperparameters
SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 128
HIDDEN_SIZE = 50
ATTENTION_SIZE = 50
GLOBAL_ATTENTION_SIZE = 50
KEEP_PROB = 0.5
BATCH_SIZE = 64 
NUM_EPOCHS = 3  
DELTA = 0.5
LAMBDA1 = 0.01
LAMBDA2 = 0.01
REG_LAMBDA = 0.01
NUM_EVAL = 1
LEARNING_RATE = 1e-3

MODEL_PATH = 'model/model.pt'
train_file = 'data/train.txt'
test_file = 'data/test.txt'
mapping_file = 'data/mapping.txt'
LR_weight_file = 'data/weight.txt'

print ("data statistics...")
SEQUENCE_LENGTH = max(get_sequence_length_total(train_file), get_sequence_length_total(test_file))
train_vocab_size, num_training_sample = get_vocabulary_size_total(train_file)
test_vocab_size, num_testing_sample = get_vocabulary_size_total(test_file)
vocabulary_size = max(train_vocab_size, test_vocab_size)


if SEQUENCE_LENGTH > 50: SEQUENCE_LENGTH = 50
l2_loss = tf.constant(0.0)

if load_pretrained_embedding:
    print("loading embeddings...")
    vocab_embedding, EMBEDDING_DIM = load_embedding('./data/embedding_128', vocabulary_size)
else:
    vocab_embedding = []

if load_LR_initializer:
    print("loading LR initializer")
    att_initializer = load_LR_weights(vocabulary_size, LR_weight_file)

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
    LR_batch_ph = tf.placeholder(tf.float32, [None, vocabulary_size], name='LR_batch_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

batch_embedded = tf.nn.dropout(batch_embedded, keep_prob_ph)

# LR model
att_w = tf.Variable(tf.truncated_normal([vocabulary_size, 1], stddev=0.1))
ATT_W, LR_loss = LR_layer(LR_batch_ph,
                          keep_prob_ph,
                          target_ph,
                          vocabulary_size,
                          LAMBDA1,
                          LAMBDA2,
                          att_w)

# (Bi-)RNN layer(-s)
rnn_outputs = RNN_layer(HIDDEN_SIZE,
                        batch_embedded,
                        seq_len_ph)

# Attention layer
attention_output, alphas, global_attention_output, betas = run_attention(rnn_outputs, 
                                                                         ATTENTION_SIZE, 
                                                                         batch_ph, 
                                                                         GLOBAL_ATTENTION_SIZE, 
                                                                         vocabulary_size, 
                                                                         ATT_W, 
                                                                         load_LR_model)

attention_output = attention_selection(rnn_outputs, 
                                       attention_output, 
                                       global_attention_output, 
                                       attention_choice, 
                                       ATTENTION_SIZE)

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    hidden_size = attention_output.shape[1].value
    W = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    if load_LR_model: loss += LR_loss
    loss += REG_LAMBDA * l2_loss
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
print ("load data...")
train_batch_generator_new = read_batch_dataset(train_file, BATCH_SIZE)
test_batch_generator_new = read_batch_dataset(test_file, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

print ("start training...")
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        tf.set_random_seed(2019)
        np.random.seed(2019)

        sess.run(tf.global_variables_initializer())
        if load_pretrained_embedding:
            sess.run(embeddings_var.assign(vocab_embedding))
        if load_LR_initializer:
            sess.run(att_w.assign(att_initializer))

        best_auc = 0
        for epoch in range(NUM_EPOCHS):
            
            loss_list,pred_list,lab_list = [],[],[]
            loss_train = 0
            loss_test = 0

            print("epoch: {}\n".format(epoch), end="")
            # Training
            num_batches = num_training_sample // BATCH_SIZE
            num_eval = num_batches // NUM_EVAL
            eval_index = 0 
            for b in tqdm(range(num_batches), ascii=True):
                x_batch, y_batch, u_batch = next(train_batch_generator_new) 
                x_batch = zero_pad(x_batch, SEQUENCE_LENGTH)
                LR_x_batch = LR_convert(x_batch, vocabulary_size)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences

                feed_dict = {batch_ph: x_batch,
                           LR_batch_ph: LR_x_batch,
                           target_ph: y_batch,
                           seq_len_ph: seq_len,
                           keep_prob_ph: KEEP_PROB
                           }

                loss_tr, lab, pred, _, summary = sess.run([loss, target_ph, y_hat, optimizer, merged], feed_dict=feed_dict)
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                loss_list.append(loss_train)

                pred_list += list(pred)
                lab_list += list(lab)
                train_writer.add_summary(summary, b + num_batches * epoch)
            
                if b % num_eval == (num_eval - 1):
                    loss_train = sum(loss_list)/len(loss_list)
                    fpr, tpr, thresholds = metrics.roc_curve(lab_list, pred_list)
                    auc_train = metrics.auc(fpr, tpr)                    
                    loss_list = []

                    num_batches_test = num_testing_sample // BATCH_SIZE
                    for c in tqdm(range(num_batches_test), ascii=True):
                        x_batch, y_batch, u_batch = next(test_batch_generator_new)
                        x_batch = zero_pad(x_batch, SEQUENCE_LENGTH)
                        LR_x_batch = LR_convert(x_batch, vocabulary_size)
                        seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences

                        feed_dict = {batch_ph: x_batch,
                                   LR_batch_ph: LR_x_batch,
                                   target_ph: y_batch,
                                   seq_len_ph: seq_len,
                                   keep_prob_ph: 1.0
                                   }

                        test_loss, lab, pred, alp, beta, summary = sess.run([loss, target_ph, y_hat, alphas, betas, merged],
                                                                 feed_dict=feed_dict) # alp: local att; beta: global att; 

                        loss_list.append(test_loss)
                        test_writer.add_summary(summary, c + (eval_index + num_eval * epoch) * num_batches_test)

                    fpr, tpr, thresholds = metrics.roc_curve(lab_list, pred_list)
                    auc_test = metrics.auc(fpr, tpr) 
                    loss_test = sum(loss_list)/len(loss_list) 
                    areas = [0.5*(tpr[i]-fpr[i]+1) for i in range(len(fpr))]
                    thr = thresholds[areas.index(max(areas))]                         

                    print("train_loss: {:.4f}, test_loss: {:.4f}, train_auc:{:.4f}, test_auc: {:.4f}".format(loss_train, loss_test, auc_train, auc_test))

                    if auc_test > best_auc:
                        best_auc = auc_test

                    eval_index += 1
                    loss_list = []

        train_writer.close()
        test_writer.close()
        
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
