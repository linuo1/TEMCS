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

        dis_coral = self.DeepCoral_loss(encoder_A, encoder_C)

        # 定义损失函数
        # loss_y = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=self.logits,
        #         labels=self.y)
        # )
        loss_y = tf.reduce_mean(
            (1.0 + self.weight) * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        self.loss = loss_y + 0.15 * dis_coral
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
            # dnn3 = tf.layers.dense(dnn1, 80,
            #                        kernel_initializer=layers.xavier_initializer(),
            #                        bias_initializer=tf.constant_initializer(0.01),
            #                        activation=tf.nn.tanh)
            # dnn3 = tf.nn.dropout(dnn3, keep_prob)
            dnn2 = tf.layers.dense(dnn1, 1,
                                   kernel_initializer=layers.xavier_initializer(),
                                   bias_initializer=tf.constant_initializer(0.01),
                                   activation=None)
            pred_A = tf.reshape(dnn2, [-1])
        return pred_A

    def DeepCoral_loss(self, source_features, target_features):
        # batch_size = source_feature.shape[0]
        weight = tf.reshape(self.weight, [-1,1])
        source_features = (1.0 + weight)*source_features
        source_batch_size = tf.cast(tf.shape(source_features)[0], tf.float32)
        target_batch_size = tf.cast(tf.shape(target_features)[0], tf.float32)
        d = tf.cast(tf.shape(source_features)[1], tf.float32)

        # Source covariance
        xm = source_features - tf.reduce_mean(source_features, 0, keep_dims=True)
        xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size

        # Target covariance
        xmt = target_features - tf.reduce_mean(target_features, 0, keep_dims=True)
        xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size
        lll = tf.multiply((xc - xct), (xc - xct))
        coral_loss = tf.reduce_sum(lll)
        coral_loss /= 4 * d * d
        return coral_loss

        # loss += tf.squeeze(tf.slice(self.weight, [i], [1])) * (kernels[s1, s2] + kernels[t1, t2])
        # loss -= tf.squeeze(tf.slice(self.weight, [i], [1])) * (kernels[s1, t2] + kernels[s2, t1])

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

