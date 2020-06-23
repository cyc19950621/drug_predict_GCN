import tensorflow.compat.v1 as tf

import numpy as np
tf.enable_eager_execution()
flags = tf.app.flags
FLAGS = flags.FLAGS


class DecagonOptimizer(object):
    def __init__(self, embeddings, latent_inters, latent_varies,
                 degrees, edge_types, edge_type2dim, placeholders,
                 margin=0.1, neg_sample_weights=1., batch_size=100):
        self.embeddings= embeddings
        self.latent_inters = latent_inters#model的中间层
        self.latent_varies = latent_varies#model的变化
        self.edge_types = edge_types#边界
        self.degrees = degrees
        self.edge_type2dim = edge_type2dim
        self.obj_type2n = {i: self.edge_type2dim[i,j][0][0] for i, j in self.edge_types}#0：500 1：400
        self.margin = margin
        self.neg_sample_weights = neg_sample_weights
        self.batch_size = batch_size

        self.inputs = placeholders['batch']#0
        self.batch_edge_type_idx = placeholders['batch_edge_type_idx']#0
        self.batch_row_edge_type = placeholders['batch_row_edge_type']#0
        self.batch_col_edge_type = placeholders['batch_col_edge_type']#0

        self.row_inputs = tf.squeeze(gather_cols(self.inputs, [0]))#→lables
        self.col_inputs = tf.squeeze(gather_cols(self.inputs, [1]))

        obj_type_n = [self.obj_type2n[i] for i in range(len(self.embeddings))]
        self.obj_type_lookup_start = tf.cumsum([0] + obj_type_n[:-1])
        self.obj_type_lookup_end = tf.cumsum(obj_type_n)

        labels = tf.reshape(tf.cast(self.row_inputs, dtype=tf.int64), [self.batch_size, 1])
        
        # 这一段是文章中的一个方法（可以先不管他）：负采样
        # estimate the model through negative sampling
        # for each drug-drug edge in graph(vi,r,vj),we sample a random edge(vi,r,vn),
        # vnis randomly choosed according to sampling distribution Pr
        neg_samples_list = []
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=labels,
                    num_true=1,
                    num_sampled=self.batch_size,
                    unique=False,
                    range_max=len(self.degrees[i][k]),
                    distortion=0.75,
                    unigrams=self.degrees[i][k].tolist())
                neg_samples_list.append(neg_samples)
        self.neg_samples = tf.cast(tf.gather(neg_samples_list, self.batch_edge_type_idx),dtype=tf.int64)# tf.int32

        self.preds = self.batch_predict(self.row_inputs, self.col_inputs)
        self.outputs = tf.diag_part(self.preds) # 返回矩阵对角线元素
        self.outputs = tf.reshape(self.outputs, [-1])#outputs输出到交叉熵损失函数
        self.neg_preds = self.batch_predict(self.neg_samples, self.col_inputs)
        self.neg_outputs = tf.diag_part(self.neg_preds)
        self.neg_outputs = tf.reshape(self.neg_outputs, [-1])
        self.predict()
        self.build()

# 这个函数很关键，是predict的核心

# predict函数输出一个矩阵相乘结果
# prediction = product3*col_embeds
# 执行预测操作，这里以c和s两种药物来举例，分别得到他们的 表示（representation）zc和zs，
# 经过预测后，可以得到他俩具有每一种关系的概率，从而得到预测结果

# 用什么公式进行的预测？#个（v,r,v) 
# 转移到untils
    def batch_predict(self, row_inputs, col_inputs):
        concatenated = tf.concat(self.embeddings, 0)#对自身的张量拼接

        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)
        row_embeds = tf.gather(row_embeds, row_inputs)
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)
        col_embeds = tf.gather(col_embeds, col_inputs)
        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)
        product1 = tf.matmul(row_embeds, latent_var)#矩阵乘法
        product2 = tf.matmul(product1, latent_inter)#矩阵乘法
        product3 = tf.matmul(product2, latent_var)#矩阵乘法
        preds = tf.matmul(product3, tf.transpose(col_embeds))
        return preds

    def predict(self):
        concatenated = tf.concat(self.embeddings, 0) #把embeddings连接到一起

        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)

        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)

        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        self.predictions = tf.matmul(product3, tf.transpose(col_embeds))

    def build(self):
        #self.cost = self._hinge_loss(self.outputs, self.neg_outputs)
        self.cost = self.xent_loss(self.outputs, self.neg_outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #  TO SOLVE: This function makes the dense tensor
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)#计算梯度

    def hinge_loss(self, aff, neg_aff):
        """Maximum-margin optimization using the hinge loss."""
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 0) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        return loss
#转移到untils
    def xent_loss(self, aff, neg_aff):
        """Cross-entropy optimization."""#交叉熵损失函数
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

#转移到untils
def gather_cols(params, indices, name=None):#为了聚集 2Dtensor 将稀疏张量聚集
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope: #需要输入一个2Dtensor和一个1Dtensor

        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])#变成一维矩阵
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],[-1, 1]) + indices, [-1])#一维矩阵
        return tf.reshape(tf.gather(p_flat, i_flat), [p_shape[0], -1])#矩阵运算