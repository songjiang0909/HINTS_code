import tensorflow as tf
from model_embedding import RGCN_embedding
from model_imputed import Impute
from model_ts import CVAE



class Model():
    def __init__(self,adj_lsit,feature_list,embedding_size,input_seq,output_seq,
                 index_table_P1P,index_table_P1A,index_table_P1V,index_table_P1K,alignment_list,lr,batch_size,beta):


        self.adj_list = adj_lsit
        self.feature_list = feature_list
        self.index_table_P1P = index_table_P1P
        self.index_table_P1A = index_table_P1A
        self.index_table_P1V = index_table_P1V
        self.index_table_P1K = index_table_P1K
        self.alignment_list = alignment_list 
        self.train_year = 5
        self.embedding_size = embedding_size
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.cvae = None
        self.impute = None
        self.embeddings = None
        self.imputed_embeddings = None
        self.loss = None
        self.align_loss = None
        self.outout_weights = None

    def build(self):
        rgcn = RGCN_embedding(self.adj_list,self.feature_list,self.alignment_list,self.train_year)
        self.embeddings,self.align_loss,self.W_1,self.W_2 = rgcn.build()
        self.impute = Impute(self.index_table_P1P,self.index_table_P1A,self.index_table_P1V,
              self.index_table_P1K,self.embeddings,self.embedding_size,self.train_year)
        self.imputed_embeddings,self.outout_weights = self.impute.build()

        self.cvae = CVAE(time_length=self.train_year,imputed_size=self.embedding_size,
                        imputed_embeds=self.imputed_embeddings,input_seq=self.input_seq,
                        output_seq=self.output_seq,batch_size = self.batch_size)


        self.cvae.build_graph()

        self.loss = tf.reduce_mean(self.cvae.pred_loss) + self.align_loss*self.beta
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
