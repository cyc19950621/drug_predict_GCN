import tensorflow.compat.v1 as tf
import numpy as np
tf.enable_eager_execution()
flags = tf.app.flags#传递参数
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def weight_variable_glorot(input_dim, output_dim, name=""):#生成均匀分布的矩阵，6/input+outdim 开方 生成-+的范围矩阵 矩阵大小为input_dim*out_dim
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def get_layer_uid(layer_name=''):#设定层ID
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

#转移到utils
def dropout_sparse(x, keep_prob, num_nonzero_elems): #对稀疏张量的防止过拟合
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)#生成随机值
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool) #向下去整 转换成bool格式
    pre_out = tf.sparse_retain(x, dropout_mask) #对X随机删除
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties 属性   
        name: String, defines the variable scope of the layer.定义层变量范围

    # Methods 方法
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')#查询词典中的name
        if not name:
            layer = self.__class__.__name__.lower() #转成小写字母
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name#设定层名，用来传递参数
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

#编码器 
class GraphConvolutionSparseMulti(MultiLayer): #图卷积稀疏多重矩阵层
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):#编码器采用relu函数激活
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name):#对每个层参数共用
            #对应原文 4.1 these architectures then share
            #  functions/parameters that define how information is shared and propagated. 
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = weight_variable_glorot(#对input output生成稀疏矩阵
                    input_dim[self.edge_type[1]], output_dim, name='weights_%d' % k)
                    #得到自己点位的信息
        #call部分是GCN的矩阵传播，通过和vars进行运算传播信息
    def _call(self, inputs):
        outputs = []
        # 有num_types种关系类型，分别计算后，append
        for k in range(self.num_types):
            x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat[self.edge_type[1]]) #防止过拟合
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)#稀疏矩阵变成密集矩阵
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)#l2归一化按行
        return outputs #outputs就是特征


class GraphConvolutionMulti(MultiLayer):#图卷积多重矩阵
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):#relu激活 编码部分
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k) #对input output生成稀疏矩阵

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' % k]) 
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)#对列表内的元素相加
        outputs = tf.nn.l2_normalize(outputs, dim=1) #dim=1 按行进行l2范化
        return outputs

#四种不同的解码器
class DEDICOMDecoder(MultiLayer): #链路预测的dedicom解码器
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = weight_variable_glorot(#对input output生成稀疏矩阵
                input_dim, input_dim, name='global_interaction')
            for k in range(self.num_types):
                tmp = weight_variable_glorot(#对input output生成稀疏矩阵
                    input_dim, 1, name='local_variation_%d' % k)
                self.vars['local_variation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.diag(self.vars['local_variation_%d' % k])
            product1 = tf.matmul(inputs_row, relation)
            product2 = tf.matmul(product1, self.vars['global_interaction'])
            product3 = tf.matmul(product2, relation)
            rec = tf.matmul(product3, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""#DistMult解码器
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                tmp = weight_variable_glorot(#对input output生成稀疏矩阵
                    input_dim, 1, name='relation_%d' % k)
                self.vars['relation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            relation = tf.diag(self.vars['relation_%d' % k])
            intermediate_product = tf.matmul(inputs_row, relation)
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""#Bilinear 解码器
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['relation_%d' % k] = weight_variable_glorot(#对input output生成稀疏矩阵
                    input_dim, input_dim, name='relation_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            intermediate_product = tf.matmul(inputs_row, self.vars['relation_%d' % k])
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction.""" # inner product解码器
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
            rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs
