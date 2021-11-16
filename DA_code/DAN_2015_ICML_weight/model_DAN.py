import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib import layers
from sklearn import metrics

class Model(object):
    def __init__(self,user_count, item_count, item_matrix, batch_size):

        self.u = tf.placeholder(tf.int32,[None,],name='user_id')  #用户id  # [B]  1行B列  B为Batch_size
        self.i = tf.placeholder(tf.int32,[None,],name='item_id')  #项目id  # [B]
        self.y = tf.placeholder(tf.float32,[None,],name='label')  #标签  # [B]

       # A可以视为books     B可以视为movies     C可以视为musics
        self.hist_A = tf.placeholder(tf.int32, [None, None], name='answer') # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        # self.hist_B = tf.placeholder(tf.int32, [None, None], name='follow')  # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        self.hist_C = tf.placeholder(tf.int32, [None, None], name='vote')  # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        self.sl_A = tf.placeholder(tf.int32, [None, ], name='item_seq_length')
        # self.sl_B = tf.placeholder(tf.int32, [None, ], name='follow_seq_length')
        self.sl_C = tf.placeholder(tf.int32, [None, ], name='vote_seq_length')  # 用户历史行为的长度   # [B]
        self.weight = tf.placeholder(tf.float32, [None, ], name='weight_sample')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.is_training = tf.placeholder(tf.bool, [])

        embedding_size = 100  #每个词语100维

        # user_emb_w = tf.get_variable('user_emb_w', [user_count, embedding_size],
        #                              initializer = tf.truncated_normal_initializer(stddev = 0.1))
        item_emb_w = tf.get_variable('item_emb_w_A', dtype=tf.float32,
                                       initializer=tf.constant(item_matrix))
        i_emb_C = tf.nn.embedding_lookup(item_emb_w, self.i)   #target ID

        # 定义特征提取器    两/三层MLP  item_emb_w_C, seq_C, embedding_size, keep_prob
        encoder_A, fir_fea_A, sec_fea_A = self.encoder_A(item_emb_w, self.hist_A, i_emb_C, embedding_size, self.keep_prob)

        encoder_C, fir_fea_C, sec_fea_C = self.encoder_C(item_emb_w, self.hist_C, i_emb_C, embedding_size, self.keep_prob)

        # 定义两个域之间的距离 与 目标域分类器 的输出
        self.logits = self.prediction_C(i_emb_C, encoder_A, encoder_C, self.keep_prob)

        # dis_AC = self.DAN_1_loss(batch_size, encoder_A, encoder_C, kernel_mul=2.0, kernel_num=5, fix_sigma=None, linear=True)
        encoder_A_list = [fir_fea_A, sec_fea_A, encoder_A]
        encoder_C_list = [fir_fea_C, sec_fea_C, encoder_C]

        dis_0 = self.DAN_1_loss(batch_size, encoder_A_list[0], encoder_C_list[0], kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                                 linear=True)
        dis_1 = self.DAN_1_loss(batch_size, encoder_A_list[1], encoder_C_list[1], kernel_mul=2.0, kernel_num=2,
                                fix_sigma=3.0, linear=True)
        dis_2 = self.DAN_1_loss(batch_size, encoder_A_list[2], encoder_C_list[2], kernel_mul=2.0, kernel_num=1,
                                fix_sigma=1.68, linear=True)
        dis_AC = dis_0 + dis_1 + dis_2
        # 定义损失函数
        # loss_y = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=self.logits,
        #         labels=self.y)
        # )
        loss_y = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        self.loss = loss_y + 0.15 * dis_AC
        # self.loss = loss_y
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)



    def encoder_A(self, item_emb_w_A, seq_A, i_emb, embedding_size, keep_prob):
        with tf.variable_scope('encoder_A'):
            # embedding_matrix_B = tf.get_variable(dtype=tf.float32, name='embedding_matrix_B',
            #                                      shape=[num_items_B, embedding_size],
            #                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            # print(embedding_matrix_B)
            embbed_seq_A = tf.nn.embedding_lookup(item_emb_w_A,
                                                  seq_A)  # embbed_seq_A=[batch_size,timestamp_A * embedding_size]
            queries = tf.expand_dims(i_emb, axis=1)
            outputs = tf.matmul(queries, tf.transpose(embbed_seq_A, [0, 2, 1]))
            outputs = outputs / (embbed_seq_A.get_shape().as_list()[-1] ** 0.5)    # Scale
            outputs = tf.nn.softmax(outputs)  # B * 1 * T   # Activation
            # Weighted Sum
            outputs = tf.matmul(outputs, embbed_seq_A)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
            fir_fea_A = tf.reshape(outputs, [-1, embedding_size])

            d_layer_2_all = tf.layers.dense(fir_fea_A, 1.5*embedding_size, activation=tf.nn.tanh, name='f2_att')
            sec_fea_A = tf.nn.dropout(d_layer_2_all, keep_prob)

            d_layer_3_all = tf.layers.dense(sec_fea_A, embedding_size, activation=None, name='f3_att')  # B*T*1
            # 为了让outputs维度和keys的维度一致
            encoder_output_A = tf.reshape(d_layer_3_all, [-1, embedding_size])  # B*1*T

        return encoder_output_A, fir_fea_A, sec_fea_A


    def encoder_C(self, item_emb_w_C, seq_C, i_emb, embedding_size, keep_prob):
        with tf.variable_scope('encoder_C'):
            embbed_seq_C = tf.nn.embedding_lookup(item_emb_w_C,
                                                  seq_C)  # embbed_seq_A=[batch_size,timestamp_A * embedding_size]
            queries = tf.expand_dims(i_emb, axis=1)
            outputs = tf.matmul(queries, tf.transpose(embbed_seq_C, [0, 2, 1]))
            outputs = outputs / (embbed_seq_C.get_shape().as_list()[-1] ** 0.5)  # Scale
            outputs = tf.nn.softmax(outputs)  # B * 1 * T   # Activation
            # Weighted Sum
            outputs = tf.matmul(outputs, embbed_seq_C)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
            fir_fea_C = tf.reshape(outputs, [-1, embedding_size])

            d_layer_2_all = tf.layers.dense(fir_fea_C, 1.5 * embedding_size, activation=tf.nn.tanh, name='f2_att')
            sec_fea_C = tf.nn.dropout(d_layer_2_all, keep_prob)

            d_layer_3_all = tf.layers.dense(sec_fea_C, embedding_size, activation=None, name='f3_att')  # B*T*1
            # 为了让outputs维度和keys的维度一致
            encoder_output_C = tf.reshape(d_layer_3_all, [-1, embedding_size])  # B*1*T

        return encoder_output_C, fir_fea_C, sec_fea_C




    def prediction_C(self, i_emb_C, encoder_A, encoder_C, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([encoder_A, encoder_C, i_emb_C], axis=-1)
            dnn1 = tf.layers.dense(concat_output, 80,
                                   kernel_initializer=layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(0.01),
                                   activation=tf.nn.tanh)
            dnn1 = tf.nn.dropout(dnn1, keep_prob)
            dnn2 = tf.layers.dense(dnn1, 1,
                                   kernel_initializer=layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(0.01),
                                   activation=None)
            pred_A = tf.reshape(dnn2, [-1])
        return pred_A

    def guassian_kernel(self, source_feature, target_feature, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # n_samples = source_feature.shape[0] + target_feature.shape[0]
        n_samples = tf.shape(source_feature)[0] + tf.shape(target_feature)[0]
        total = tf.concat([source_feature, target_feature], axis=0)
        xx = tf.expand_dims(total, axis=0)
        print("n_samples", n_samples)
        total0 = tf.tile(xx, [n_samples, 1, 1])
        total1 = tf.tile(tf.expand_dims(total, axis=1), [1, n_samples, 1])
        L2_distance = tf.reduce_sum(((total0 - total1) ** 2), axis=-1)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = tf.reduce_sum(L2_distance) / tf.cast((n_samples ** 2 - n_samples), dtype=tf.float32)
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return tf.reduce_sum(kernel_val, axis=0)



    def DAN_1_loss(self, batch_size, source_feature, target_feature, kernel_mul=2.0, kernel_num=5, fix_sigma=None, linear=False):
        # batch_size = source_feature.shape[0]
        kernels = self.guassian_kernel(source_feature, target_feature,
                                       kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if linear:
            loss = 0
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                # loss += kernels[s1, s2] + kernels[t1, t2]
                # loss -= kernels[s1, t2] + kernels[s2, t1]
                loss += tf.squeeze(tf.slice(self.weight, [i], [1])) * (kernels[s1, s2] + kernels[t1, t2])
                loss -= tf.squeeze(tf.slice(self.weight, [i], [1])) * (kernels[s1, t2] + kernels[s2, t1])
            DAN_loss = loss / tf.cast(batch_size, dtype=tf.float32)
        else:
            loss1 = 0
            for s1 in range(batch_size):
                for s2 in range(s1 + 1, batch_size):
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    loss1 += kernels[s1, s2] + kernels[t1, t2]
            loss1 = loss1 / tf.cast((batch_size * (batch_size - 1) / 2), dtype=tf.float32)

            loss2 = 0
            for s1 in range(batch_size):
                for s2 in range(batch_size):
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    loss2 -= kernels[s1, t2] + kernels[s2, t1]
            loss2 = loss2 / tf.cast((batch_size * batch_size), dtype=tf.float32)
            DAN_loss = loss1 + loss2
        return DAN_loss


    def train(self,sess, uij, lr, keep_prob):
        y_pred, loss,_ = sess.run([self.logits, self.loss, self.train_op],feed_dict={
            self.u : uij[0],
            self.i : uij[1],
            self.y : uij[2],

            self.hist_A: uij[4],
            # self.hist_B: uij[5],
            self.hist_C: uij[3],

            self.sl_A: uij[7],
            # self.sl_B: uij[8],
            self.sl_C: uij[6],
            self.weight: uij[9],

            self.lr : lr,
            self.keep_prob : keep_prob,
            self.is_training: True
        })

        return y_pred, loss


    def eval(self, sess, uij, lr, keep_prob):
        res1 = sess.run(self.logits, feed_dict={
            self.u: uij[0],
            self.i: uij[1],

            self.hist_A: uij[4],
            # self.hist_B: uij[5],
            self.hist_C: uij[3],

            self.sl_A: uij[7],
            # self.sl_B: uij[8],
            self.sl_C: uij[6],

            self.lr: lr,
            self.keep_prob: keep_prob,
            self.is_training: False
        })
        res2 = sess.run(self.logits, feed_dict={
            self.u: uij[0],
            self.i: uij[2],

            self.hist_A: uij[4],
            # self.hist_B: uij[5],
            self.hist_C: uij[3],

            self.sl_A: uij[7],
            # self.sl_B: uij[8],
            self.sl_C: uij[6],

            self.lr: lr,
            self.keep_prob: keep_prob,
            self.is_training: False
        })
        return np.mean(res1 - res2 > 0)

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

