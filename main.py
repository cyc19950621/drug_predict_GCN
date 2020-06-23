# 可参考这篇博客，写了具体算法原理
#  https://blog.csdn.net/weixin_41075215/article/details/104104846?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
# 文件结构：
# main.py是模型整体运行一次的流程：生成数据-——建模（model,optimizer）——训练
# 训练结果用main.py开头的 get accuracy scores函数进行计算，得到评价指标 roc aupr apk50

# 模型构建在model.py
# gcn层在layers.py
# 编码器和解码器在layer.py
# 模型优化器在optimizer.py 模型预测功能使用optimizer.prediction函数

# 模型整体概览：
# 模型搭建 model.py
# 模型训练主要使用 optimizer, 在optimizer.py
# 核心的结果输出在：get_accuracy_scores函数中，这一句：rec = sess.run(opt.predictions, feed_dict=feed_dict)
# 输出的rec矩阵是一个500*500维矩阵（为什么不是400*400维？药物数量在这里是400，基因数量是500）

from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os
import tensorflow.compat.v1 as tf

import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
tf.disable_eager_execution()
# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################

# 输出模型结果的函数
# 这个函数很重要，输出的东西通过这个函数作用，得到roc,apk50等评价参数

    #1/（1+e·*-x) 得到学习后的评分概率
def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)
# rec是一个500*500的矩阵！
# 表示两个药物具有每一种关系的概率
# 如何进行的predict,参看opimizer.py的predict函数
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))
    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1
    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'
        predicted.append((score, edge_ind))
        edge_ind += 1
    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]
    roc_sc = metrics.roc_auc_score(labels_all, preds_all) #
    aupr_sc = metrics.average_precision_score(labels_all, preds_all) #
    apk_sc = rank_metrics.apk(actual, predicted, k=50) #apk50 average precision at 50%
    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):#构建形成稀疏张量
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32, shape=(), name='degress'),
        'dropout': tf.placeholder_with_default(0. , shape=(), name='dropout'),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####

val_test_size = 0.05
n_genes = 500
n_drugs = 400
n_drugdrug_rel_types = 3
gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)
# 作为NP-hard问题的图的二分问题就是将一个无向图分为两个相同大小的组而是的他们跨组的边的数量最小。
# 更一般地，图的l-分割问题就是将一个无向图分割成l个相同大小的组从而使得组与组之间的边最小。
# 这里，我认为是随机产生一个有链接的图
# 生成50个组 每个组10个定点 点和点之间的连接概率0.2 组 0.05

gene_adj = nx.adjacency_matrix(gene_net) #500*500的邻接矩阵
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze() #基因之间 求和 删除一个纬度 之后的链接度 500维向量

gene_drug_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))#500*400
# scr_matrix是把一个稀疏的np.array压缩
drug_gene_adj = gene_drug_adj.transpose(copy=True)#400*500 转向 药物和基因的稀疏矩阵

drug_drug_adj_list = [] #400*400*3
tmp = np.dot(drug_gene_adj, gene_drug_adj)
for i in range(n_drugdrug_rel_types):
    mat = np.zeros((n_drugs, n_drugs)) #
    for d1, d2 in combinations(list(range(n_drugs)), 2):
        if tmp[d1, d2] == i + 4:
            mat[d1, d2] = mat[d2, d1] = 1.
    drug_drug_adj_list.append(sp.csr_matrix(mat))
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

# 400*3 每种关系下，drug的连接度

# data representation
#两个词典  对应不同的矩阵
#1 基因和药物类型和连接词典
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],#药物连接矩阵和药物
}
#2 基因和药物的度的词典 两个array
degrees = {
    0: [gene_degrees, gene_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

# featureless (genes)
gene_feat = sp.identity(n_genes) #生成500*500矩阵
gene_nonzero_feat, gene_num_feat = gene_feat.shape #（500，500）
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())#转换为元组

# features (drugs)
drug_feat = sp.identity(n_drugs)
drug_nonzero_feat, drug_num_feat = drug_feat.shape # （400，400）
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {
    0: gene_num_feat,#0：500
    1: drug_num_feat,#0：400
}
nonzero_feat = {
    0: gene_nonzero_feat,#0 500
    1: drug_nonzero_feat,#0 400
}
feat = {
    0: gene_feat, 
    1: drug_feat,
}

# edge_types都是字典变量
edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
# 根据数据类型不同，解码器不同
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}#对词典遍历 确定连接
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)
# >edge_types.keys()
# >dict_keys([(0, 0), (0, 1), (1, 0), (1, 1)])
# >edge_types.values()
# >dict_values([2, 1, 1, 6])

###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150#每150显示进度

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

# optimizer.py
print("Create optimizer")
with tf.name_scope('optimizer'):
    #opt作为decagon的优化器
    opt = DecagonOptimizer(
        embeddings=model.embeddings,#decagon model.embedding
        latent_inters=model.latent_inters,#decagon model.latent_inters
        latent_varies=model.latent_varies,#decagon model.latent_varies
        degrees=degrees,#(dict)
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,#传参函数
        batch_size=FLAGS.batch_size,#512
        margin=FLAGS.max_margin)#0.1
        #损失函数
    

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())#初始化
feed_dict = {}#传参词典
#初始化
###########################################################
#
# Train model
#
###########################################################

print("Train model")
####################################
for epoch in range(FLAGS.epochs):
    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()
#############################问题部分
        # Training step: run single weight update
        outs = sess.run((opt.opt_op, opt.cost, opt.batch_edge_type_idx), feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]
#################################
        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
