import numpy as np
from scipy.special import erf
import tensorflow as tf
from utils import *

class CVAE():
    def __init__(self,time_length,imputed_size,imputed_embeds,input_seq,output_seq,batch_size):
        self.imputed_size = imputed_size
        self.batch_size = batch_size
        self.time_length = time_length
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.imputed_embeds = imputed_embeds
        self.eta = None
        self.mu = None
        self.sigma = None
        self.pred_loss = None


    def log_normal(self,eta,mu,sigma,t):
        return eta*1/(np.sqrt(2*np.pi)*sigma*t) * tf.exp(-(np.log(t)-mu)**2/(2*sigma**2))

    def dashun_model (self,eta,mu,sigma,t):
        x = (np.log(t)-mu)/(1+sigma)
        norm = tf.contrib.distributions.Normal(loc=0., scale=1.)
        inte = norm.cdf(x)
        return tf.math.exp(eta*inte)-1


    def build_graph(self):


        mlp_hidden_unit = {"encoder_1":50,"encoder_2":10,
                           "decoder_1":20,"decoder_2":8,"decoder_3":1,
                           "rnn":50,"conditional_1":20}


        cell = tf.nn.rnn_cell.GRUCell(mlp_hidden_unit["rnn"])
        initial_state = cell.zero_state(batch_size=self.batch_size,dtype=tf.float32)
        with tf.variable_scope("decoder",reuse=False):
            #conditional
            _, self.final_state = tf.nn.dynamic_rnn(cell, self.imputed_embeds, initial_state=initial_state)
            h1_imputed_emb_mean = tf.layers.dense(self.final_state, mlp_hidden_unit["conditional_1"],
                                                  activation=tf.nn.relu, name="condition_mean_1")
            self.condition_mean = tf.layers.dense(h1_imputed_emb_mean, mlp_hidden_unit["encoder_2"],
                                                  name="condition_mean_2")
            self.z = self.condition_mean
            ##decoder
            h1_decoder_eta = tf.layers.dense(self.z,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="eta_decoder1")
            h2_decoder_eta = tf.layers.dense(h1_decoder_eta, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="eta_decoder2")
            self.eta = tf.layers.dense(h2_decoder_eta,mlp_hidden_unit["decoder_3"],name="eta_decoder3")

            h1_decoder_mu = tf.layers.dense(self.z,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="mu_decoder1")
            h2_decoder_mu = tf.layers.dense(h1_decoder_mu, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="mu_decoder2")
            self.mu = tf.layers.dense(h2_decoder_mu,mlp_hidden_unit["decoder_3"],name="mu_decoder3")

            h1_decoder_sigma = tf.layers.dense(self.z,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="sigma_decoder1")
            h2_decoder_sigma = tf.layers.dense(h1_decoder_sigma, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="sigma_decoder2")
            self.sigma = tf.layers.dense(h2_decoder_sigma,mlp_hidden_unit["decoder_3"],name="sigma_decoder3")

            self.citation_pred_list = [self.dashun_model(self.eta, self.mu, self.sigma, t) for t in range(1, 6)]
            self.citation_pred = tf.transpose(tf.reshape(self.citation_pred_list,[5,self.batch_size]))

        self.pred_loss = tf.reduce_sum(tf.square(self.citation_pred-self.output_seq),1)



        with tf.variable_scope("decoder",reuse=True):
            _, self.final_state_test = tf.nn.dynamic_rnn(cell, self.imputed_embeds, initial_state=initial_state)
            h1_imputed_emb_mean_test = tf.layers.dense(self.final_state_test, mlp_hidden_unit["conditional_1"],
                                                  activation=tf.nn.relu, name="condition_mean_1")
            self.condition_mean_test = tf.layers.dense(h1_imputed_emb_mean_test, mlp_hidden_unit["encoder_2"],
                                                  name="condition_mean_2")

            self.z_test = self.condition_mean_test

            h1_decoder_eta_test = tf.layers.dense(self.z_test,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="eta_decoder1")
            h2_decoder_eta_test = tf.layers.dense(h1_decoder_eta_test, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="eta_decoder2")
            self.eta_test = tf.layers.dense(h2_decoder_eta_test,mlp_hidden_unit["decoder_3"],name="eta_decoder3")

            h1_decoder_mu_test = tf.layers.dense(self.z_test,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="mu_decoder1")
            h2_decoder_mu_test = tf.layers.dense(h1_decoder_mu_test, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="mu_decoder2")
            self.mu_test = tf.layers.dense(h2_decoder_mu_test,mlp_hidden_unit["decoder_3"],name="mu_decoder3")

            h1_decoder_sigma_test = tf.layers.dense(self.z_test,mlp_hidden_unit["decoder_1"],activation=tf.nn.relu,name="sigma_decoder1")
            h2_decoder_sigma_test = tf.layers.dense(h1_decoder_sigma_test, mlp_hidden_unit["decoder_2"], activation=tf.nn.relu, name="sigma_decoder2")
            self.sigma_test = tf.layers.dense(h2_decoder_sigma_test,mlp_hidden_unit["decoder_3"],name="sigma_decoder3")

            self.citation_pred_list_test = [self.dashun_model(self.eta_test, self.mu_test, self.sigma_test, t) for t in range(1, 6)]
            self.citation_pred_test = tf.transpose(tf.reshape(self.citation_pred_list_test,[5,self.batch_size]))




