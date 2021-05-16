import tensorflow as tf

class Impute():##Needs 2005N  2000~2004 P1P P1A P1V P1K
    def __init__(self,index_table_P1P,index_table_P1A,index_table_P1V,index_table_P1K,embeddings,embedding_size,train_year):
        self.index_table_P1P = index_table_P1P 
        self.index_table_P1A = index_table_P1A
        self.index_table_P1V = index_table_P1V
        self.index_table_P1K = index_table_P1K
        self.embedding = embeddings
        self.embedding_size = embedding_size
        self.train_year = train_year
        self.realtion_weight = {'P1P_weight':tf.get_variable(name='P1P_weight',dtype=tf.float32,shape=()),
                                 'P1A_weight': tf.get_variable(name='P1A_weight', dtype=tf.float32,shape=()),
                                 'P1V_weight':tf.get_variable(name='P1V_weight',dtype=tf.float32,shape=()),
                                 'P1K_weight':tf.get_variable(name='P1K_weight',dtype=tf.float32,shape=())}
        self.outout_weights = []


    def build(self):
        imputed_embeddings = []
        for i in range(self.train_year):
            output_P1P = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1P[i])
            output_mean_P1P_sum = tf.reduce_sum(output_P1P, 1)
            idx_P1P = tf.to_float(tf.math.greater_equal(self.index_table_P1P[i], 0))
            idx_P1P = tf.reduce_sum(idx_P1P, 1, keepdims=True)
            output_mean_P1P = output_mean_P1P_sum / (idx_P1P+0.001)
            self.output_mean_P1P = tf.multiply(output_mean_P1P, self.realtion_weight['P1P_weight'])

            output_P1A = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1A[i])
            output_mean_P1A_sum = tf.reduce_sum(output_P1A, 1)
            idx_P1A = tf.to_float(tf.math.greater_equal(self.index_table_P1A[i], 0))
            idx_P1A = tf.reduce_sum(idx_P1A, 1, keepdims=True)
            output_mean_P1A = output_mean_P1A_sum / (idx_P1A+0.001)
            self.output_mean_P1A = tf.multiply(output_mean_P1A, self.realtion_weight['P1A_weight'])

            output_P1V = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1V[i])
            output_mean_P1V_sum = tf.reduce_sum(output_P1V, 1)
            idx_P1V = tf.to_float(tf.math.greater_equal(self.index_table_P1V[i], 0))
            idx_P1V = tf.reduce_sum(idx_P1V, 1, keepdims=True)
            output_mean_P1V = output_mean_P1V_sum / (idx_P1V+0.001)
            self.output_mean_P1V = tf.multiply(output_mean_P1V, self.realtion_weight['P1V_weight'])

            output_P1K = tf.nn.embedding_lookup(self.embedding[i], ids=self.index_table_P1K[i])
            output_mean_P1K_sum = tf.reduce_sum(output_P1K, 1)
            idx_P1K = tf.to_float(tf.math.greater_equal(self.index_table_P1K[i], 0))
            idx_P1K = tf.reduce_sum(idx_P1K, 1, keepdims=True)
            output_mean_P1K = output_mean_P1K_sum / (idx_P1K+0.001)
            self.output_mean_P1K = tf.multiply(output_mean_P1K, self.realtion_weight['P1K_weight'])

            output_mean = [output_mean_P1P,output_mean_P1A,output_mean_P1V,output_mean_P1K]
            imputed_embeddings.append(tf.reduce_mean(output_mean, 0))
        imputed_embeddings = tf.concat(imputed_embeddings, axis=1)
        imputed_embeddings = tf.reshape(imputed_embeddings,shape=(-1,self.train_year,self.embedding_size))  
        self.outout_weights = [self.realtion_weight['P1P_weight'],self.realtion_weight['P1A_weight'],
                               self.realtion_weight['P1V_weight'],self.realtion_weight['P1K_weight']]

        return imputed_embeddings,self.outout_weights
