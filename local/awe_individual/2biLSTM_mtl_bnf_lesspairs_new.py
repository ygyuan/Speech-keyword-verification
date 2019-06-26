#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write all matrices in a Kaldi archive to Numpy .npz format.

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

import numpy as np
import random
import tensorflow as tf
import scipy.spatial.distance as distance
import data_processing_mtl_bnf_new as dp                      #################changed#####################
from scipy.spatial.distance import pdist
np.set_printoptions(threshold=np.inf)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class SimpleLSTM(object):
    def __init__(self, is_training, config):
        hidden_size = config.hidden_size
        input_size = config.input_size
        margin = config.margin
        lr = config.learning_rate
        kp = config.keep_prob
        output_size=config.output_size

        # Layer1 x1
        self._input_x1 = input_x1 = tf.placeholder(tf.float32, [None, None, input_size])
        self._input_x1_lengths = input_x1_lengths = tf.placeholder(tf.int32, [None])
        input_x1_lengths_64 = tf.to_int64(input_x1_lengths)

        l2r_cell_layer1 = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        r2l_cell_layer1 = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        if is_training and kp < 1:
            r2l_cell_layer1 = tf.contrib.rnn.DropoutWrapper(r2l_cell_layer1, output_keep_prob=kp)
            l2r_cell_layer1 = tf.contrib.rnn.DropoutWrapper(l2r_cell_layer1, output_keep_prob=kp)

        with tf.variable_scope('l2r_layer1'):
            l2r_outputs_layer1, _ = tf.nn.dynamic_rnn(l2r_cell_layer1, input_x1, dtype=tf.float32,sequence_length=input_x1_lengths )
        with tf.variable_scope('r2l_layer1'):
            r2l_outputs_layer1, _ = tf.nn.dynamic_rnn(r2l_cell_layer1,tf.reverse_sequence(input_x1, input_x1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x1_lengths)

        r2l_outputs_layer1 = tf.reverse_sequence(r2l_outputs_layer1, input_x1_lengths_64, 1)
        input_x1_layer2 = tf.concat([l2r_outputs_layer1, r2l_outputs_layer1],2, 'concat_layer1_x1')

        #layer 2 x1
        l2r_cell_layer2 = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        r2l_cell_layer2 = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        #if is_training and kp < 1:
        #    r2l_cell_layer2 = tf.contrib.rnn.DropoutWrapper(r2l_cell_layer2, output_keep_prob=kp)
        #    l2r_cell_layer2 = tf.contrib.rnn.DropoutWrapper(l2r_cell_layer2, output_keep_prob=kp)
        with tf.variable_scope('l2r_layer2'):
            l2r_outputs_layer2, _ = tf.nn.dynamic_rnn(l2r_cell_layer2, input_x1_layer2, dtype=tf.float32,
                                                            sequence_length=input_x1_lengths)
        with tf.variable_scope('r2l_layer2'):
            r2l_outputs_layer2, _ = tf.nn.dynamic_rnn(r2l_cell_layer2, tf.reverse_sequence(input_x1_layer2, input_x1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x1_lengths)

        l2r_outputs = tf.gather(tf.reshape(tf.concat(l2r_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x1)[0]) * tf.shape(input_x1)[1] + input_x1_lengths - 1)
        r2l_outputs = tf.gather(tf.reshape(tf.concat(r2l_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x1)[0]) * tf.shape(input_x1)[1] + input_x1_lengths - 1)

        x1_output=tf.concat([l2r_outputs, r2l_outputs],1, 'concat_x1')

        if output_size>0:
            with tf.variable_scope('output'):
                x1_output = tf.layers.dense(inputs=x1_output, units=output_size, activation=tf.nn.tanh)

        self._final_state = x1 = self.normalization(x1_output)

        if not is_training:
            return

        # input_x2
        # Layer 1 x2
        self._input_x2 = input_x2 = tf.placeholder(tf.float32, [None, None, input_size])
        self._input_x2_lengths = input_x2_lengths = tf.placeholder(tf.int32, [None])
        input_x2_lengths_64 = tf.to_int64(input_x2_lengths)

        #if is_training and kp < 1:
        #    r2l_cell_layer1 = tf.contrib.rnn.DropoutWrapper(r2l_cell_layer1, output_keep_prob=kp)
        #    l2r_cell_layer1 = tf.contrib.rnn.DropoutWrapper(l2r_cell_layer1, output_keep_prob=kp)

        with tf.variable_scope('l2r_layer1', reuse=True):
            l2r_outputs_layer1, _ = tf.nn.dynamic_rnn(l2r_cell_layer1, input_x2, dtype=tf.float32,
                                                            sequence_length=input_x2_lengths)

        with tf.variable_scope('r2l_layer1', reuse=True):
            r2l_outputs_layer1, _ = tf.nn.dynamic_rnn(r2l_cell_layer1,tf.reverse_sequence(input_x2, input_x2_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x2_lengths)
        r2l_outputs_layer1 = tf.reverse_sequence(r2l_outputs_layer1, input_x2_lengths_64, 1)

        # Layer 2 x2
        input_x2_layer2 = tf.concat([l2r_outputs_layer1, r2l_outputs_layer1],2, 'concat_layer1_x2')
        with tf.variable_scope('l2r_layer2', reuse=True):
            l2r_outputs_layer2, _ = tf.nn.dynamic_rnn(l2r_cell_layer2, input_x2_layer2,dtype=tf.float32,
                                                            sequence_length=input_x2_lengths)
        with tf.variable_scope('r2l_layer2', reuse=True):
            r2l_outputs_layer2, _ = tf.nn.dynamic_rnn(r2l_cell_layer2, tf.reverse_sequence(input_x2_layer2,input_x2_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x2_lengths)

        l2r_outputs = tf.gather(tf.reshape(tf.concat(l2r_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x2)[0]) * tf.shape(input_x2)[
                                          1] + input_x2_lengths - 1)
        r2l_outputs = tf.gather(tf.reshape(tf.concat(r2l_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x2)[0]) * tf.shape(input_x2)[
                                          1] + input_x2_lengths - 1)
        x2_output=tf.concat([l2r_outputs, r2l_outputs],1, 'concat_x2')

        if output_size>0:
            with tf.variable_scope('output', reuse=True):
                x2_output = tf.layers.dense(inputs=x2_output, units=output_size, activation=tf.nn.tanh)

        x2 = self.normalization(x2_output)

        # Layer 1 x3
        self._input_x3 = input_x3 = tf.placeholder(tf.float32, [None, None, input_size])
        self._input_x3_lengths = input_x3_lengths = tf.placeholder(tf.int32, [None])
        input_x3_lengths_64 = tf.to_int64(input_x3_lengths)

        with tf.variable_scope('l2r_layer1', reuse=True):
            l2r_outputs_layer1, _ = tf.nn.dynamic_rnn(l2r_cell_layer1, input_x3, dtype=tf.float32,
                                                            sequence_length=input_x3_lengths)
        with tf.variable_scope('r2l_layer1', reuse=True):
            r2l_outputs_layer1, _ = tf.nn.dynamic_rnn(r2l_cell_layer1, tf.reverse_sequence(input_x3, input_x2_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x3_lengths)

        r2l_outputs_layer1 = tf.reverse_sequence(r2l_outputs_layer1, input_x3_lengths_64, 1)
        input_x3_layer2 = tf.concat([l2r_outputs_layer1, r2l_outputs_layer1],2, 'concat_layer1_x3')

        with tf.variable_scope('l2r_layer2', reuse=True):
            l2r_outputs_layer2, _ = tf.nn.dynamic_rnn(l2r_cell_layer2, input_x3_layer2, dtype=tf.float32,
                                                            sequence_length=input_x3_lengths)

        with tf.variable_scope('r2l_layer2', reuse=True):
            r2l_outputs_layer2, _ = tf.nn.dynamic_rnn(r2l_cell_layer2, tf.reverse_sequence(input_x3_layer2,input_x3_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x3_lengths)

        l2r_outputs = tf.gather(tf.reshape(tf.concat(l2r_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x3)[0]) * tf.shape(input_x3)[
                                          1] + input_x3_lengths - 1)
        r2l_outputs = tf.gather(tf.reshape(tf.concat(r2l_outputs_layer2,1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x3)[0]) * tf.shape(input_x3)[
                                          1] + input_x3_lengths - 1)
        x3_output=tf.concat([l2r_outputs, r2l_outputs],1, 'concat_x3')

        if output_size>0:
            with tf.variable_scope('output', reuse=True):
                x3_output = tf.layers.dense(inputs=x3_output, units=output_size, activation=tf.nn.tanh)

        x3 = self.normalization(x3_output)

        self.sim, self.dis, self.absolute_loss, self.contract_loss = self.contrastive_loss(margin, x1, x2, x3)

        self.loss = 10.0*(self.contract_loss+self.absolute_loss) #+ self.regularization_cost

        self._train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def contrastive_loss(self, margin, x1, x2, x3):
        sim = tf.multiply(x1, x2)
        sim = tf.reduce_sum(sim, 1)
        dis = tf.multiply(x1, x3)
        dis = tf.reduce_sum(dis, 1)

        loss1 = tf.maximum(margin+dis-sim,0)
        loss2 = tf.maximum( sim-4*margin-dis,0)
        loss3 = (tf.maximum( 0.4-sim,0)+tf.maximum(dis-0.4,0))
        return tf.reduce_mean(sim), tf.reduce_mean(dis), tf.reduce_mean(loss1)+tf.reduce_mean(loss2), tf.reduce_mean(loss3)
        #return tf.reduce_mean(sim), tf.reduce_mean(dis), tf.reduce_mean(loss1), tf.reduce_mean(loss3)

    def normalization(self, x):
        norm = tf.sqrt(tf.reduce_sum(tf.square(x), 1, keep_dims=True) + 1e-8)
        return x / tf.tile(norm, [1, tf.shape(x)[1]])

    @property
    def input_x1(self):
        return self._input_x1

    @property
    def input_x2(self):
        return self._input_x2

    @property
    def input_x3(self):
        return self._input_x3

    @property
    def train_step(self):
        return self._train_step

    @property
    def input_x1_lengths(self):
        return self._input_x1_lengths

    @property
    def input_x2_lengths(self):
        return self._input_x2_lengths

    @property
    def input_x3_lengths(self):
        return self._input_x3_lengths

    @property
    def final_state(self):
        return self._final_state

    @property
    def word_state(self):
        return self._word_state


class BatchIteratorTriplets(object):
    """In every epoch the tokens for the different-pairs are sampled."""

    def __init__(self, conf, data_dir, train_labels, train_matches, sample_every_epoch=True, each_same_pairs=10):
        """
        If `n_same_pairs` is given, this number of same pairs is sampled,
        otherwise all same pairs are used.
        """
        self.rng = conf.rng
        self.train_labels = ["_".join(i.split("_")[:-3]) for i in train_labels ]
        pos_matches, neg_matches = train_matches
        self.pos_matrix = distance.squareform(pos_matches)
        self.neg_matrix = distance.squareform(neg_matches)
        self.sample_every_epoch = sample_every_epoch
        self.each_same_pairs = each_same_pairs
        print self.pos_matrix.shape, self.neg_matrix.shape

    def _sample_indices(self):
        x1_same_indices = []
        x2_same_indices = []
        x3_diff_indices = []
        for i in xrange(len(self.pos_matrix)):
            #n_pairs = min(self.each_same_pairs,len(np.where( self.pos_matrix[i]==True)[0]), len(np.where( self.neg_matrix[i]==True)[0]))
            temp=min(len(np.where( self.pos_matrix[i]==True)[0]), len(np.where( self.neg_matrix[i]==True)[0]))
            if temp<self.each_same_pairs:
                continue
            n_pairs=int(1.0*self.each_same_pairs/np.log10([temp])[0])
            same_sample = self.rng.choice( np.where( self.pos_matrix[i] == True)[0], size=n_pairs, replace=False)
            diff_sample = self.rng.choice( np.where( self.neg_matrix[i] == True)[0], size=n_pairs, replace=False)
            for j in xrange(n_pairs):
                x1_same_indices.append(i)
                x2_same_indices.append(same_sample[j])
                x3_diff_indices.append(diff_sample[j])

        x1_same_indices = np.array(x1_same_indices, dtype=np.int32)
        x2_same_indices = np.array(x2_same_indices, dtype=np.int32)
        x3_diff_indices = np.array(x3_diff_indices, dtype=np.int32)
        print len(x1_same_indices),x1_same_indices[:10],[self.train_labels[i] for i in x1_same_indices[:10]]
        print len(x2_same_indices),x2_same_indices[:10],[self.train_labels[i] for i in x2_same_indices[:10]]
        print len(x3_diff_indices),x3_diff_indices[:10],[self.train_labels[i] for i in x3_diff_indices[:10]]
        return x1_same_indices, x2_same_indices, x3_diff_indices

    def __iter__(self):
        # Sample different tokens for this epoch
        if self.sample_every_epoch:
            self.x1_same_indices, self.x2_same_indices, self.x3_diff_indices = self._sample_indices()
        yield (self.x1_same_indices, self.x2_same_indices, self.x3_diff_indices)

def run_epoch(m, op, sess, batch_size, train_data, train_lengths, train_batch_iterator):
    cost = 0
    sim = 0.0
    dis = 0.0
    for batch in train_batch_iterator:
        input_x1_set, input_x2_set, input_x3_set = batch
        num_mini_batches = len(input_x1_set) / batch_size
        print len(input_x1_set), batch_size, num_mini_batches
        for i_batch in xrange(num_mini_batches):
            sim, dis, loss, _ = sess.run([m.sim, m.dis, m.loss, op], {m.input_x1: train_data[input_x1_set[i_batch*batch_size: (i_batch + 1)*batch_size]],
                                              m.input_x2: train_data[input_x2_set[i_batch*batch_size: (i_batch + 1)*batch_size]],
                                              m.input_x3: train_data[input_x3_set[i_batch*batch_size: (i_batch + 1)*batch_size]],
                                              m.input_x1_lengths: train_lengths[input_x1_set[i_batch*batch_size: (i_batch + 1)*batch_size]],
                                              m.input_x2_lengths: train_lengths[input_x2_set[i_batch*batch_size: (i_batch + 1)*batch_size]],
                                              m.input_x3_lengths: train_lengths[input_x3_set[i_batch*batch_size: (i_batch + 1)*batch_size]]})
            print "i_batch: ", i_batch, "; sim:", sim, "; dis: ",dis, "; Loss: ",loss
            #print "i_batch: ", i_batch, "; Loss: ",loss
            cost += loss
        return cost/num_mini_batches, sim, dis #, mean_cost/num_mini_batches

def eval_embeddings(m, sess, data, lengths, config):
    batch_size = 500
    embeddings = []
    num_mini_batches = len(data)/batch_size
    print data.shape, batch_size, num_mini_batches
    for i_batch in xrange(num_mini_batches):
        embedding = sess.run(m.final_state, {m.input_x1: data[i_batch*batch_size: (i_batch + 1)*batch_size],
                                            m.input_x1_lengths: lengths[i_batch*batch_size: (i_batch + 1)*batch_size]})
        embeddings.append(embedding)

    if len(data)-num_mini_batches*batch_size > 0:
        embedding = sess.run(m.final_state, {m.input_x1: data[num_mini_batches*batch_size:],
                                             m.input_x1_lengths: lengths[num_mini_batches*batch_size:]})
        embeddings.append(embedding)
    return np.vstack(embeddings)

def eval_train(train_embeddings, train_matches, train_labels, thresholds, config):
    dist = pdist(train_embeddings, 'cosine')
    precision = [i.split("_")[-4] for i in train_labels ]

    pos_matches, neg_matches = train_matches
    pos_matrix = distance.squareform(pos_matches)
    neg_matrix = distance.squareform(neg_matches)
    dist_matrix = distance.squareform(dist)
    print len(precision),pos_matrix.shape,neg_matrix.shape,dist_matrix.shape

    pos_result=0.0
    neg_result=0.0
    ave_result=0.0
    threshold_result=0.0
    for threshold in thresholds:
        new_threshold=1-threshold
        neg_precision=[]
        pos_precision=[]
        for i in xrange(len(precision)):
            if precision[i]=="1":
                temp_pos = dist_matrix[i][pos_matrix[i]==True]
                temp=np.mean( np.array(temp_pos) <= new_threshold)
                pos_precision.append(temp)
            else:
                temp_neg = dist_matrix[i][neg_matrix[i]==True]
                temp=np.mean( np.array(temp_neg) > new_threshold)
                neg_precision.append(temp)
        pos_precision=np.array(pos_precision)
        neg_precision=np.array(neg_precision)
        pos=np.mean(pos_precision >= 0.5)
        neg=np.mean(neg_precision >= 0.5)
        result=(pos+neg)/2.0
        if ave_result<=result+1e-5:
            threshold_result=threshold
            pos_result=pos
            neg_result=neg
            ave_result=result
        print "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                threshold, pos_result, neg_result, ave_result, pos, neg, result)
    return threshold_result, pos_result, neg_result, ave_result

def eval_test(valid_embeddings, valid_labels, train_embeddings, train_labels, thresholds, config):
    dis_positive=[]
    pos_labels_seg={}
    pos_ave_distance={}
    min_templates=1000
    for i in xrange(len(valid_labels)):
        word_i="_".join(valid_labels[i].split("_")[:-4])+"_1"
        if word_i not in pos_labels_seg:
            pos_labels=[j for j in train_labels if j.startswith(word_i)]
            if len(pos_labels)>11:
                pos_labels=random.sample(pos_labels,11)
            pos_labels_seg[word_i]=[train_labels.index(j) for j in pos_labels]
            if len(pos_labels_seg[word_i])<min_templates:
                min_templates=len(pos_labels_seg[word_i])
                print word_i,min_templates
            lengths=len(pos_labels_seg[word_i])
            ave_distance=[]
            for x in xrange(lengths-1):
                for y in xrange(x+1,lengths):
                    ave_distance.append(np.array([np.sum(np.multiply(train_embeddings[pos_labels_seg[word_i][x]],train_embeddings[pos_labels_seg[word_i][y]]))]))
            pos_ave_distance[word_i]= np.mean(ave_distance)
        dis_positive.append(np.array([np.sum(np.multiply(valid_embeddings[i],train_embeddings[j])) for j in pos_labels_seg[word_i] ]))
    print len(dis_positive),len(dis_positive[0])
    precision=[i.split("_")[-4] for i in valid_labels ]

    pos_result=0.0
    neg_result=0.0
    ave_result=0.0
    threshold_result=0.0
    for threshold in thresholds:
        neg_precision=[]
        pos_precision=[]
        for i in xrange(len(valid_labels)):
            word_i="_".join(valid_labels[i].split("_")[:-4])+"_1"
            if precision[i]=="1":
                temp=np.mean(np.array(dis_positive[i]) >= threshold + pos_ave_distance[word_i])
                pos_precision.append(temp)
            else:
                temp = np.mean(np.array(dis_positive[i]) < threshold + pos_ave_distance[word_i])
                neg_precision.append(temp)
        pos_precision=np.array(pos_precision)
        neg_precision=np.array(neg_precision)
        pos=np.mean(pos_precision >= 0.5)
        neg=np.mean(neg_precision >= 0.5)
        result=(pos+neg)/2.0
        if ave_result<=result+1e-5:
            threshold_result=threshold
            pos_result=pos
            neg_result=neg
            ave_result=result
        print "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                threshold, pos_result, neg_result, ave_result, pos, neg, result)
    return threshold_result, pos_result, neg_result, ave_result

def eval_test2(valid_embeddings, valid_labels, train_embeddings, train_labels, thresholds, config):

    dis_positive=[]
    pos_labels_seg={}
    pos_ave_distance={}
    for i in xrange(len(valid_labels)):
        word_i="_".join(valid_labels[i].split("_")[:-4])+"_1"
        if word_i not in pos_labels_seg:
            pos_labels=[j for j in train_labels if j.startswith(word_i)]
            if len(pos_labels)>400:
                pos_labels=random.sample(pos_labels,400)
            pos_labels_seg[word_i]=[train_labels.index(j) for j in pos_labels]
            lengths=len(pos_labels_seg[word_i])
            ave_distance=[]
            for x in xrange(lengths-1):
                for y in xrange(x+1,lengths):
                    ave_distance.append(np.array([np.sum(np.multiply(train_embeddings[pos_labels_seg[word_i][x]],train_embeddings[pos_labels_seg[word_i][y]]))]))
            pos_ave_distance[word_i]= np.mean(ave_distance)
        dis_positive.append(np.array([np.sum(np.multiply(valid_embeddings[i],train_embeddings[j])) for j in pos_labels_seg[word_i] ]))
    print len(dis_positive), len(dis_positive[0])
    precision=[i.split("_")[-4] for i in valid_labels ]

    result_file="wiki_NowOnline_20181005_20181115/20181221_20181224.result_new"
    pos_result=0.0
    neg_result=0.0
    ave_result=0.0
    threshold_result=0.0
    for threshold in thresholds:
        count=0
        result_file1=open("20181221_20181224.result6_"+str(threshold),"w")
        neg_precision=[]
        pos_precision=[]
        for line in open(result_file):
            count+=1
            line=unicode(line, "utf-8").split("\n")[0].split(" ")
            uttid=line[0].split("/")[-1].split(".")[0]+"-"+str(count)
            precision=line[-1].split("\r")[0].encode('utf-8')

            index=-1
            for i in xrange(len(valid_labels)):
                if valid_labels[i].endswith(uttid):
                    index=i
                    break
            if index<0:
                print index
                quit()
            word_i="_".join(valid_labels[index].split("_")[:-4])+"_1"
            temp=np.mean(np.array(dis_positive[index]) >= threshold + pos_ave_distance[word_i])
            if precision=="正确":
                pos_precision.append(temp>=0.5)
                print "正确", pos_precision[-1], temp #, line[-3].encode('utf-8'), len(dis_positive[index]), np.histogram(dis_positive[index])
            elif precision=="错误":
                neg_precision.append(temp<0.5)
                print "错误", neg_precision[-1], temp #,line[-3].encode('utf-8'), len(dis_positive[index]), np.histogram(dis_positive[index])
            else:
                print precision
                quit()
            if temp >= 0.5:
                if precision=="正确":
                    result_file1.write(" 1.0"+" 正确 "+"{:.3f}".format(temp)+ "\n")
                else:
                    result_file1.write(" 0.0"+" 正确 "+"{:.3f}".format(temp)+ "\n")
            else:
                if precision=="正确":
                    result_file1.write(" 1.0"+" 错误 "+"{:.3f}".format(temp)+ "\n")
                else:
                    result_file1.write(" 0.0"+" 错误 "+"{:.3f}".format(temp)+ "\n")
        pos=np.mean(pos_precision)
        neg=np.mean(neg_precision)
        result=(pos+neg)/2.0
        if ave_result<=result+1e-5:
            threshold_result=threshold
            pos_result=pos
            neg_result=neg
            ave_result=result
        print "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(threshold, pos_result, neg_result, ave_result, pos, neg, result)
    return threshold_result, pos_result, neg_result, ave_result

class Config(object):
    input_size = 43 #############changed####################
    eval_batch_size = 490

    def __init__(self, hs, m, lr, kp, bs, epo, inits, sets, n_same_pairs, output_size, gpu_device):
        self.init_scale = float(inits)
        self.epoch = int(epo)
        self.batch_size = int(bs)
        self.hidden_size = int(hs)
        self.margin = float(m)
        self.learning_rate = float(lr)
        self.keep_prob = float(kp)
        self.sets=int(sets)
        self.n_same_pairs=int(n_same_pairs)
        self.output_size=int(output_size)
        self.gpu_device=gpu_device
        self.rng = np.random.RandomState(42)

def main():
    # parse the arguments, then pass them to configuration
    args = dp.ArgsHandle()
    print args
    conf = Config(args.hs, args.m, args.lr, args.kp, args.bs, args.epo, args.inits, args.sets, args.n_same_pairs, args.output_size, args.gpu_device)
    print conf
    # define the model we are gonna load, see if we want to train from scratch
    last_epoch = int(args.lastepoch)
    continue_training = False
    if last_epoch > -1:
        continue_training = True
        model_name = dp.ModelName(args)

    # pad input_x1 and input_c1, get validation data ready
    train = np.load(os.path.join(args.data_dir,"train.npz"))
    dev = np.load(os.path.join(args.data_dir,"dev.npz"))
    train_data, train_lengths, train_matches, train_labels = dp.GetTestData(train,True)
    print np.array(train_data).shape, np.array(train_lengths).shape, np.array(train_labels).shape
    valid_data, valid_lengths, valid_labels = dp.GetTestData(dev, False)
    print np.array(valid_data).shape, np.array(valid_lengths).shape, np.array(valid_labels).shape

    #quit()
    # Make batch iterators
    train_batch_iterator = BatchIteratorTriplets(
        conf, args.data_dir, train_labels, train_matches, sample_every_epoch=True
        )
    # initialize conputing graphs, get ready for saving models
    initializer = tf.random_uniform_initializer(-conf.init_scale, conf.init_scale)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        m = SimpleLSTM(True, conf)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
        mvalid = SimpleLSTM(False, conf)

    #parameter_count=np.sum([np.prod(v.shape) for v in tf.trainable_variables()])/(1024*1024)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_device
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=18)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # train the ModelNamels
    with tf.Session(config=config)as sess:
        if continue_training:
            saver.restore(sess, os.path.join(args.model_dir,model_name))
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(last_epoch+1, conf.epoch + 1):
            output_name = dp.OutputName(args)
            fout = open(os.path.join(args.output_dir,output_name+'.txt'), 'a+')
            avg_cost, sim, dis = run_epoch(m, m.train_step, sess, conf.batch_size, train_data, train_lengths, train_batch_iterator)
            print i, avg_cost, sim, dis
            fout.write('Epoch ' + str(i) + ': ' + str(avg_cost)+" " +str(sim)+" "+str(dis)+ '\n')
            # calculate AP every 5 epochs
            if (i+1) % 5 == 0 or i == 0:
                train_embeddings = eval_embeddings(mvalid, sess, train_data, train_lengths, conf)
                valid_embeddings = eval_embeddings(mvalid, sess, valid_data, valid_lengths, conf)
                print valid_embeddings.shape, train_embeddings.shape
                thresholds=np.arange(-0.10, 0.10, 0.010)
                threshold, pos_precision, neg_precision, dev_ap = eval_test(valid_embeddings, valid_labels, train_embeddings, train_labels, thresholds, conf)
                fout.write('Dev AP: ' + str(dev_ap) +" " + str(pos_precision)+" "+str(neg_precision)+" " +str(threshold)+ '\n')
                fout.write('Train AP: ' + str(dev_ap) +" " + str(pos_precision)+" "+str(neg_precision)+" " +str(threshold)+ '\n')
                fout.close()
                best_AP, best_idx = dp.BestAP(os.path.join(args.output_dir,output_name+'.txt'))
                print "i, dev_ap, best_AP, best_idx:", i, dev_ap, best_AP, best_idx
                if ( dev_ap >= best_AP-0.00001 or i==conf.epoch or i==99 or i==199 or i==299 or i==399):
                    saver.save(sess, os.path.join(args.model_dir,output_name + '-' + str(i)))
                    print "best index:", best_idx
                    keep_indices = [ 99, 199, 299, 399, best_idx, i, conf.epoch]
                    print "keep_indices, output_name:", keep_indices, output_name
                    dp.ModelClean(keep_indices,args.model_dir, output_name)
if __name__ == '__main__':
    main()
