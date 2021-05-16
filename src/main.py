from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import time
import argparse
from tqdm import tqdm

from model import Model
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=700,
                    help='Number of epochs to train.')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='embedding vector dimension')
parser.add_argument('--batch_size', type=int, default=3000,
                    help='batch_size')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default="aminer",
                    help='Choose dataset')
args = parser.parse_args()


def train(sess, model, feed):
    sess.run(model.optimizer, feed_dict=feed)
    loss = sess.run(model.loss, feed_dict=feed)
    print("training_loss:{}".format(loss))



def main(beta,num):
    tf.reset_default_graph()
    embedding_size = args.embedding_size


    adj_1_P1P = tf.sparse_placeholder(name='adj_1_P1P', shape=(None, None), dtype=tf.float32)
    adj_1_P1A = tf.sparse_placeholder(name='adj_1_P1A', shape=(None, None), dtype=tf.float32)
    adj_1_P1V = tf.sparse_placeholder(name='adj_1_P1V', shape=(None, None), dtype=tf.float32)
    adj_1_P1K = tf.sparse_placeholder(name='adj_1_P1K', shape=(None, None), dtype=tf.float32)
    adj_1_self = tf.sparse_placeholder(name='adj_1_self', shape=(None, None), dtype=tf.float32)
    adj_1 = [adj_1_P1P,adj_1_P1A,adj_1_P1V,adj_1_P1K,adj_1_self]
    feature_1 = tf.placeholder(name='feature_1', shape=(None, 4), dtype=tf.float32)


    adj_2_P1P = tf.sparse_placeholder(name='adj_2_P1P', shape=(None, None), dtype=tf.float32)
    adj_2_P1A = tf.sparse_placeholder(name='adj_2_P1A', shape=(None, None), dtype=tf.float32)
    adj_2_P1V = tf.sparse_placeholder(name='adj_2_P1V', shape=(None, None), dtype=tf.float32)
    adj_2_P1K = tf.sparse_placeholder(name='adj_2_P1K', shape=(None, None), dtype=tf.float32)
    adj_2_self = tf.sparse_placeholder(name='adj_2_self', shape=(None, None), dtype=tf.float32)
    adj_2 = [adj_2_P1P,adj_2_P1A,adj_2_P1V,adj_2_P1K,adj_2_self]
    feature_2 = tf.placeholder(name='feature_2', shape=(None, 4), dtype=tf.float32)


    adj_3_P1P = tf.sparse_placeholder(name='adj_3_P1P', shape=(None, None), dtype=tf.float32)
    adj_3_P1A = tf.sparse_placeholder(name='adj_3_P1A', shape=(None, None), dtype=tf.float32)
    adj_3_P1V = tf.sparse_placeholder(name='adj_3_P1V', shape=(None, None), dtype=tf.float32)
    adj_3_P1K = tf.sparse_placeholder(name='adj_3_P1K', shape=(None, None), dtype=tf.float32)
    adj_3_self = tf.sparse_placeholder(name='adj_3_self', shape=(None, None), dtype=tf.float32)
    adj_3 = [adj_3_P1P,adj_3_P1A,adj_3_P1V,adj_3_P1K,adj_3_self]
    feature_3 = tf.placeholder(name='feature_3', shape=(None, 4), dtype=tf.float32)


    adj_4_P1P = tf.sparse_placeholder(name='adj_4_P1P', shape=(None, None), dtype=tf.float32)
    adj_4_P1A = tf.sparse_placeholder(name='adj_4_P1A', shape=(None, None), dtype=tf.float32)
    adj_4_P1V = tf.sparse_placeholder(name='adj_4_P1V', shape=(None, None), dtype=tf.float32)
    adj_4_P1K = tf.sparse_placeholder(name='adj_4_P1K', shape=(None, None), dtype=tf.float32)
    adj_4_self = tf.sparse_placeholder(name='adj_4_self', shape=(None, None), dtype=tf.float32)
    adj_4 = [adj_4_P1P,adj_4_P1A,adj_4_P1V,adj_4_P1K,adj_4_self]
    feature_4 = tf.placeholder(name='feature_4', shape=(None, 4), dtype=tf.float32)


    adj_5_P1P = tf.sparse_placeholder(name='adj_5_P1P', shape=(None, None), dtype=tf.float32)
    adj_5_P1A = tf.sparse_placeholder(name='adj_5_P1A', shape=(None, None), dtype=tf.float32)
    adj_5_P1V = tf.sparse_placeholder(name='adj_5_P1V', shape=(None, None), dtype=tf.float32)
    adj_5_P1K = tf.sparse_placeholder(name='adj_5_P1K', shape=(None, None), dtype=tf.float32)
    adj_5_self = tf.sparse_placeholder(name='adj_5_self', shape=(None, None), dtype=tf.float32)
    adj_5 = [adj_5_P1P,adj_5_P1A,adj_5_P1V,adj_5_P1K,adj_5_self]
    feature_5 = tf.placeholder(name='feature_5', shape=(None, 4), dtype=tf.float32)




    adj_list = [adj_1,adj_2,adj_3,adj_4,adj_5]
    feature_list = [feature_1,feature_2,feature_3,feature_4,feature_5]
    input_seq = tf.placeholder(name="input_citation", shape=(None, 5), dtype=tf.float32)
    output_seq = tf.placeholder(name="output_citation", shape=(None, 5), dtype=tf.float32)
    index_table_P1P = tf.placeholder(name='index_P1P', shape=(5,None, None), dtype=tf.int32)
    index_table_P1A = tf.placeholder(name='index_P1A', shape=(5,None, None), dtype=tf.int32)
    index_table_P1V = tf.placeholder(name='index_P1A', shape=(5,None, None), dtype=tf.int32)
    index_table_P1K = tf.placeholder(name='index_P1A', shape=(5,None, None), dtype=tf.int32)


    alignment_list_1 = tf.placeholder(name='alignment_list1',shape=(2,None),dtype=tf.int32)
    alignment_list_2 = tf.placeholder(name='alignment_list2', shape=(2, None), dtype=tf.int32)
    alignment_list_3 = tf.placeholder(name='alignment_list3', shape=(2, None), dtype=tf.int32)
    alignment_list_4 = tf.placeholder(name='alignment_list4', shape=(2, None), dtype=tf.int32)
    alignment_list = [alignment_list_1,alignment_list_2,alignment_list_3,alignment_list_4]




    model = Model(adj_list, feature_list, embedding_size, input_seq, output_seq,
                  index_table_P1P, index_table_P1A, index_table_P1V, index_table_P1K,alignment_list,args.lr,args.batch_size,beta)
    model.build()


    if args.dataset == "aps":
        adj_list_, feature_list_, labels, index_table_P1P_, index_table_P1A_, index_table_P1V_,\
        index_table_P1K_,alignment_list_,label_max, label_min,rank = load_data_aps(2000,'train')

    if args.dataset == "aminer":
       adj_list_, feature_list_, labels, index_table_P1P_, index_table_P1A_, index_table_P1V_,\
       index_table_P1K_,alignment_list_,label_max, label_min = load_data_aminer(2005,'train')

    minibatches = generate_batch(args.batch_size,labels, index_table_P1P_, index_table_P1A_,
                   index_table_P1V_,index_table_P1K_)
    feed_dict_base = {
        adj_1_P1P : adj_list_[0][0],
        adj_1_P1A : adj_list_[0][1],
        adj_1_P1V : adj_list_[0][2],
        adj_1_P1K : adj_list_[0][3],
        adj_1_self : adj_list_[0][4],

        adj_2_P1P: adj_list_[1][0],
        adj_2_P1A: adj_list_[1][1],
        adj_2_P1V: adj_list_[1][2],
        adj_2_P1K: adj_list_[1][3],
        adj_2_self: adj_list_[1][4],

        adj_3_P1P: adj_list_[2][0],
        adj_3_P1A: adj_list_[2][1],
        adj_3_P1V: adj_list_[2][2],
        adj_3_P1K: adj_list_[2][3],
        adj_3_self: adj_list_[2][4],

        adj_4_P1P: adj_list_[3][0],
        adj_4_P1A: adj_list_[3][1],
        adj_4_P1V: adj_list_[3][2],
        adj_4_P1K: adj_list_[3][3],
        adj_4_self: adj_list_[3][4],

        adj_5_P1P: adj_list_[4][0],
        adj_5_P1A: adj_list_[4][1],
        adj_5_P1V: adj_list_[4][2],
        adj_5_P1K: adj_list_[4][3],
        adj_5_self: adj_list_[4][4],

        feature_1: feature_list_[0],
        feature_2: feature_list_[1],
        feature_3: feature_list_[2],
        feature_4: feature_list_[3],
        feature_5: feature_list_[4],

        alignment_list_1: alignment_list_[0],
        alignment_list_2: alignment_list_[1],
        alignment_list_3: alignment_list_[2],
        alignment_list_4: alignment_list_[3],
    }


    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(args.epochs):
            print("epoch:{}".format(epoch))
            for idx, list in enumerate(minibatches):
                print("batch:{}/{},epoch:{}".format(idx,len(minibatches)-1,epoch))
                feed = feed_dict_base
                feed.update({
                    input_seq: list[0],
                    output_seq: list[0],
                    index_table_P1P: list[1],
                    index_table_P1A: list[2],
                    index_table_P1V: list[3],
                    index_table_P1K: list[4],
                })

                train(sess, model, feed)
        print("training done!!")



        if args.dataset == "aps":
            adj_list_, feature_list_, labels, index_table_P1P_, index_table_P1A_, index_table_P1V_, 
            index_table_P1K_, alignment_list_, label_max, label_min,rank = load_data_aps(2005,'test')


        if args.dataset == "aminer":
           adj_list_, feature_list_, labels, index_table_P1P_, index_table_P1A_, index_table_P1V_, \
           index_table_P1K_, alignment_list_, label_max, label_min = load_data_aminer(2010,'test')


        minibatches_test = generate_batch(args.batch_size, labels, index_table_P1P_, index_table_P1A_,
                                     index_table_P1V_, index_table_P1K_)
        ##test by batch
        pred_train = []
        pred_test = []
        for idx, list in enumerate(minibatches_test):
            print("batch:{}/{} test".format(idx, len(minibatches_test)))
            feed_dict_base = {
                                                adj_1_P1P : adj_list_[0][0],
                                                adj_1_P1A : adj_list_[0][1],
                                                adj_1_P1V : adj_list_[0][2],
                                                adj_1_P1K : adj_list_[0][3],
                                                adj_1_self : adj_list_[0][4],

                                                adj_2_P1P: adj_list_[1][0],
                                                adj_2_P1A: adj_list_[1][1],
                                                adj_2_P1V: adj_list_[1][2],
                                                adj_2_P1K: adj_list_[1][3],
                                                adj_2_self: adj_list_[1][4],

                                                adj_3_P1P: adj_list_[2][0],
                                                adj_3_P1A: adj_list_[2][1],
                                                adj_3_P1V: adj_list_[2][2],
                                                adj_3_P1K: adj_list_[2][3],
                                                adj_3_self: adj_list_[2][4],

                                                adj_4_P1P: adj_list_[3][0],
                                                adj_4_P1A: adj_list_[3][1],
                                                adj_4_P1V: adj_list_[3][2],
                                                adj_4_P1K: adj_list_[3][3],
                                                adj_4_self: adj_list_[3][4],

                                                adj_5_P1P: adj_list_[4][0],
                                                adj_5_P1A: adj_list_[4][1],
                                                adj_5_P1V: adj_list_[4][2],
                                                adj_5_P1K: adj_list_[4][3],
                                                adj_5_self: adj_list_[4][4],

                                                feature_1: feature_list_[0],
                                                feature_2: feature_list_[1],
                                                feature_3: feature_list_[2],
                                                feature_4: feature_list_[3],
                                                feature_5: feature_list_[4],

                                                alignment_list_1: alignment_list_[0],
                                                alignment_list_2: alignment_list_[1],
                                                alignment_list_3: alignment_list_[2],
                                                alignment_list_4: alignment_list_[3],
            }
            feed = feed_dict_base
            feed.update({
                input_seq: list[0],
                output_seq: list[0],
                index_table_P1P: list[1],
                index_table_P1A: list[2],
                index_table_P1V: list[3],
                index_table_P1K: list[4],
            })
            pred_train_batch = sess.run(model.cvae.citation_pred, feed_dict=feed)
            pred_train_batch = label_recover(pred_train_batch, label_max, label_min)
            pred_train.append(pred_train_batch)
            pred_test_batch = sess.run(model.cvae.citation_pred_test, feed_dict=feed)
            pred_test_batch = label_recover(pred_test_batch, label_max, label_min)
            pred_test.append(pred_test_batch)


        pred_train = np.concatenate(pred_train)
        pred_test = np.concatenate(pred_test)
        print (np.array(pred_train).shape)
        print (np.array(pred_test).shape)
        labels = label_recover(labels,label_max, label_min)

        np.savetxt("../result/"+args.dataset+"_test_test_beta_"+str(beta)+"num_"+str(num)+".txt", pred_test, fmt="%.2f %.2f %.2f %.2f %.2f")
        np.savetxt("../result/"+args.dataset+"_test_labels_"+str(beta)+"num_"+str(num)+".txt", labels, fmt="%.2f %.2f %.2f %.2f %.2f")

    #    cal_metric(args.batch_size,len(minibatches_test))




if __name__ == '__main__':
    t = time.time()
    beta = [0.5]
    num = [0,1,2,3,4,5,6,7,8,9]

    for i in tqdm(range(len(beta))):
        b = beta[i]
        for n in num:
            print (b,n)
            main(b,n)
    print ('time:{}'.format((time.time()-t)/60))
