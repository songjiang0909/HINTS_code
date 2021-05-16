import numpy as np
import tensorflow as tf
from rgcn import RGCN


class RGCN_embedding():
    def __init__(self, adj_list,feature_list,alignment_index,train_year):
        self.adj_list = adj_list
        self.feature_list = feature_list
        self.alignment_index = alignment_index  
        self.train_year = train_year
        self.out_dims = {
            "out_dim1":64,
            "out_dim2":128,
        }
        self.rgcn_object = None

    def build(self):
        embeddings = []
        for i in range(self.train_year):
            self.rgcn_object = RGCN(self.adj_list[i],self.feature_list[i],self.out_dims,i)
            embeddings.append(self.rgcn_object.call())


        align_loss = 0
        for i in range(self.train_year-1):
            align_embeds1 = tf.nn.embedding_lookup(embeddings[i], ids=self.alignment_index[i][0])
            align_embeds2 = tf.nn.embedding_lookup(embeddings[i+1], ids=self.alignment_index[i][1])
            align_loss = align_loss + (1/(self.train_year-1))*tf.norm((align_embeds1 - align_embeds2), ord='euclidean') / tf.to_float(
                tf.shape(align_embeds1)[0])


        return  embeddings,align_loss,self.rgcn_object.W_1,self.rgcn_object.W_2,


