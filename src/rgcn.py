from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import  tensorflow as tf

class RGCN():
    def __init__(self,adj,feature,out_dims,graph_order,dropout=False):
        self.adj = adj
        self.feature = feature
        self.input_dim = self.feature.shape[1]
        self.out_dims = out_dims
        self.relation = ['P1P','P1A','P1V','P1K','self']
        self.support = len(self.relation)
        self.dropout = dropout
        self.graph_order = graph_order
        self.W_1 = None
        self.W_2 = None
        self.activation = tf.nn.relu

        self.build()

    def add_weight(self,relation,layer_number,input_dim,out_dim):
        return tf.get_variable(shape = (input_dim,out_dim), name='graph_order_{}_{}_W_{}'.format(self.graph_order,relation,layer_number))

    def build(self):
        features_shape = tf.shape(self.feature)
        self.W_1 = tf.concat([self.add_weight(relation=self.relation[i],layer_number=1,
                                              input_dim=self.input_dim,out_dim=self.out_dims["out_dim1"])
                                               for i in range(self.support)],0)
        self.W_2 = tf.concat([self.add_weight(relation=self.relation[i],layer_number=2,
                                              input_dim=self.out_dims["out_dim1"],out_dim=self.out_dims["out_dim2"])
                                               for i in range(self.support)],0)
    def call(self): 
        supports = []
        for i in range(self.support):
            supports.append(tf.sparse.sparse_dense_matmul(self.adj[i],self.feature))
        supports = tf.concat(supports,1)
        H_1 = tf.matmul(supports,self.W_1)
        H_1 = self.activation(H_1)
        H_1 = tf.nn.dropout(H_1, 0.2)

        supports = []
        for i in range(self.support):
            supports.append(tf.sparse.sparse_dense_matmul(self.adj[i],H_1))
        supports = tf.concat(supports,1)
        H_2 = tf.matmul(supports,self.W_2)

        return  H_2














