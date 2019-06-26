#!/usr/bin/env python

"""
Apply embeddings

Author: Yougen Yuan
Contact: ygyuan@nwpu-aslp.org
Date: 2019
"""

biLSTM2=__import__("2biLSTM_mtl_bnf_lesspairs_new")
import numpy as np
import tensorflow as tf
import data_processing_mtl_bnf_new as dp                      #################changed#####################
np.set_printoptions(threshold=np.inf)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main():
    # parse the arguments, then pass them to configuration
    args = dp.ArgsHandle()
    print args
    conf = biLSTM2.Config(args.hs, args.m, args.lr, args.kp, args.bs, args.epo, args.inits, args.sets, args.n_same_pairs, args.output_size, args.gpu_device)
    print conf
    model_name = dp.ModelName(args)
    train = np.load(os.path.join(args.data_dir,"train.npz"))
    dev = np.load(os.path.join(args.data_dir,"dev.npz"))

    train_data, train_lengths, train_matches, train_labels = dp.GetTestData(train, True)
    print np.array(train_data).shape, np.array(train_lengths).shape, np.array(train_labels).shape
    valid_data, valid_lengths, valid_labels = dp.GetTestData(dev, False)
    print np.array(valid_data).shape, np.array(valid_lengths).shape, np.array(valid_labels).shape
    with tf.variable_scope('model'):
        mvalid = biLSTM2.SimpleLSTM(False, conf)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_device
    saver = tf.train.Saver(tf.global_variables())

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # train the ModelNamels
    with tf.Session(config=config)as sess:
        saver.restore(sess, os.path.join(args.model_dir,model_name))
        valid_embeddings = biLSTM2.eval_embeddings(mvalid, sess, valid_data, valid_lengths, conf)
        train_embeddings = biLSTM2.eval_embeddings(mvalid, sess, train_data, train_lengths, conf)
        thresholds=np.arange(-0.07-0.10, -0.07+0.10, 0.01)
        threshold, pos_precision, neg_precision, dev_ap = biLSTM2.eval_test(valid_embeddings, valid_labels, train_embeddings, train_labels, thresholds, conf)
        print dev_ap
if __name__ == '__main__':
    main()
