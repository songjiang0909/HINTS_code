import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
import pickle as pkl
from sklearn.metrics import mean_absolute_error,mean_squared_error


def round_list2D(lst):
    lst = np.array(lst)
    return np.around(lst,2)

def linear_normolization(feature):
    maximum = feature.max()
    minimum = feature.min()
    return (feature-minimum)/(maximum-minimum)

def lable_normorlization(labels):
    maximum = labels.max()
    minimum = labels.min()
    new_value = (labels-minimum)/(maximum-minimum)
    return new_value,maximum,minimum

def label_recover(values,label_max,label_min):
    return values*(label_max-label_min)+label_min

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)
def build_symmetric_adj(adj_):
    return [(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)) for adj in adj_]


def load_data_aminer(year,flag):

    with open('../aminer/individual_data/graph_' + str(year - 5) + '_nf.pkl', 'rb') as f:
        graph_1 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_1['adj']]
        adj_1 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_1 = graph_1['feature']

    with open('../aminer/individual_data/graph_' + str(year - 4) + '_nf.pkl', 'rb') as f:
        graph_2 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_2['adj']]
        adj_2 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_2 = graph_2['feature']
    
    with open('../aminer/individual_data/graph_' + str(year - 3) + '_nf.pkl', 'rb') as f:
        graph_3 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_3['adj']]
        adj_3 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_3 = graph_3['feature']

    with open('../aminer/individual_data/graph_' + str(year - 2) + '_nf.pkl', 'rb') as f:
        graph_4 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_4['adj']]
        adj_4 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_4 = graph_4['feature']
 
    with open('../aminer/individual_data/graph_' + str(year - 1) + '_nf.pkl', 'rb') as f:
        graph_5 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_5['adj']]
        adj_5 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_5 = graph_5['feature']

    adj_list = [adj_1,adj_2,adj_3,adj_4,adj_5]
    feature_list=[feature_1,feature_2,feature_3,feature_4,feature_5]

    if flag == "train":
        with open('../aminer/select_index_train.pkl','rb') as f:
            rank = pkl.load(f)

    elif flag == "test":
        with open('../aminer/select_index_test.pkl','rb') as f:
            rank = pkl.load(f)

    with open('../aminer/cumulative_log_labels_new.pkl','rb') as f:
        labels = pkl.load(f)['P' + str(year) + '_label'].iloc[rank, 1:6].values

    labels, label_max, label_min = lable_normorlization(labels)
    with open('../aminer/individual_data/index_' + str(year) + '.pkl', 'rb') as f:
        index = pkl.load(f)
        index_table_P1P = index[0][:,rank,:]
        index_table_P1A = index[1][:,rank,:]
        index_table_P1V = index[2][:,rank,:]
        index_table_P1K = index[3][:,rank,:]


    with open('../aminer/individual_data/alignment_nodes.pkl','rb')as f:
        alignment_ids = pkl.load(f)
    alignment_index = [alignment_ids['aligment_id_'+str(year-(5-i))] for i in range(4)]

    print ("load {} data done!!".format(year))
    print ("There are totally {} papers".format(len(labels)))
    return adj_list,feature_list,labels,index_table_P1P,index_table_P1A,index_table_P1V,\
           index_table_P1K,alignment_index,label_max, label_min

def generate_batch(batch_size,labels,index_table_P1P,index_table_P1A,index_table_P1V,index_table_P1K):
    number_batch = int(len(labels)/batch_size)
    minibatches = []
    for i in range(number_batch):
        label = labels[i*batch_size:(i+1)*batch_size,:]
        index_P = [index[i*batch_size:(i+1)*batch_size,:] for index in index_table_P1P]
        index_A = [index[i * batch_size:(i + 1) * batch_size, :] for index in index_table_P1A]
        index_V = [index[i * batch_size:(i + 1) * batch_size, :] for index in index_table_P1V]
        index_K = [index[i * batch_size:(i + 1) * batch_size, :] for index in index_table_P1K]
        minibatches.append([label,index_P,index_A,index_V,index_K])
    print('minibatch partition done!!')
    return  minibatches

def cal_metric(batch_size,batch_number):
    y_true = np.loadtxt('../result/test_labels.txt')
    y_true = y_true[:batch_size*batch_number,:]
    y_pred = np.loadtxt('../result/test_test.txt')
    y_pred = np.maximum(y_pred,0)

    print ('MAE:{}'.format(mean_absolute_error(y_true,y_pred)))
    print ('RMSE:{}'.format(np.sqrt(mean_squared_error(y_pred,y_true))))


def load_data_aps(year,flag):

    with open('../aps/individual_graph_' + str(year - 5) + '_nf.pkl', 'rb') as f:
        graph_1 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_1['adj']]
        adj_1 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_1 = graph_1['feature']

    with open('../aps/individual_graph_' + str(year - 4) + '_nf.pkl', 'rb') as f:
        graph_2 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_2['adj']]
        adj_2 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_2 = graph_2['feature']
   
    with open('../aps/individual_graph_' + str(year - 3) + '_nf.pkl', 'rb') as f:
        graph_3 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_3['adj']]
        adj_3 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_3 = graph_3['feature']
   
    with open('../aps/individual_graph_' + str(year - 2) + '_nf.pkl', 'rb') as f:
        graph_4 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_4['adj']]
        adj_4 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_4 = graph_4['feature']
    
    with open('../aps/individual_graph_' + str(year - 1) + '_nf.pkl', 'rb') as f:
        graph_5 = pkl.load(f)
        adj_ = [normalize(adj_)for adj_ in graph_5['adj']]
        adj_5 = [convert_sparse_matrix_to_sparse_tensor(adj) for adj in adj_]
        feature_5 = graph_5['feature']

    adj_list = [adj_1,adj_2,adj_3,adj_4,adj_5]
    feature_list=[feature_1,feature_2,feature_3,feature_4,feature_5]

    if flag == "train":
        with open('../aps/select_index_train.pkl','rb') as f:
            rank = pkl.load(f)

    elif flag == "test":
        with open('../aps/select_index_test.pkl','rb') as f:
            rank = pkl.load(f)


    labels = pd.read_csv('../aps/cumulative_log_labels'+str(year)+'.txt').iloc[rank, 1:6].values
    print ("label: {}".format(labels.shape))

    labels, label_max, label_min = lable_normorlization(labels)
    with open('../aps/index_' + str(year) + '.pkl', 'rb') as f:
        index = pkl.load(f)
        index_table_P1P = index[0][:,rank,:]
        index_table_P1A = index[1][:,rank,:]
        index_table_P1V = index[2][:,rank,:]
        index_table_P1K = index[3][:,rank,:]


    with open('../aps/alignment_nodes.pkl','rb')as f:
        alignment_ids = pkl.load(f)
    alignment_index = [alignment_ids['aligment_id_'+str(year-(5-i))] for i in range(4)]

    print ("load {} data done!!".format(year))
    print ("There are totally {} papers".format(len(labels)))
    return adj_list,feature_list,labels,index_table_P1P,index_table_P1A,index_table_P1V,\
           index_table_P1K,alignment_index,label_max, label_min,rank









