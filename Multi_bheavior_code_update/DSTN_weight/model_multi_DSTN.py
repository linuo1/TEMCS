import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib import layers

class Model(object):
    def __init__(self,user_count,item_count, item_matrix):

        self.u = tf.placeholder(tf.int32,[None,],name='user_id')  #用户id  # [B]  1行B列  B为Batch_size
        self.i = tf.placeholder(tf.int32,[None,],name='item_id')  #项目id  # [B]
        self.y = tf.placeholder(tf.float32,[None,],name='label')  #标签  # [B]

        # self.item_matrix = tf.placeholder(tf.float32, [None, 300], name = 'item_all_feature') #所有的item的特征  [item_count, 300]

        self.hist_it = tf.placeholder(tf.int32, [None, None], name='answer') # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        self.hist_follow = tf.placeholder(tf.int32, [None, None], name='follow')  # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        self.hist_vote = tf.placeholder(tf.int32, [None, None], name='vote')  # 用户历史回答过得一些问题  # [B, T]  T为用户历史行为的最大长度
        self.sl = tf.placeholder(tf.int32, [None, ], name='item_seq_length')
        self.sl_follow = tf.placeholder(tf.int32, [None, ], name='follow_seq_length')
        self.sl_vote = tf.placeholder(tf.int32, [None, ], name='vote_seq_length')  # 用户历史行为的长度   # [B]
        self.weight_f = tf.placeholder(tf.float32, [None, ], name='weight_sample_f')
        self.weight_v = tf.placeholder(tf.float32, [None, ], name='weight_sample_v')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.is_training = tf.placeholder(tf.bool, [])

        embedding_size =100  #每个词语100维

        # user_emb_w = tf.get_variable('user_emb_w', [user_count, embedding_size],
        #                              initializer = tf.truncated_normal_initializer(stddev = 0.1))
        item_emb_w = tf.get_variable('item_emb_w', dtype=tf.float32,
                                     initializer=tf.constant(item_matrix))
        item_emb_all = item_emb_w

        i_emb = tf.nn.embedding_lookup(item_emb_all, self.i)
        # u_p_emb = tf.nn.embedding_lookup(user_emb_w, self.u)


        # 融合用户的过去的历史 LSTM建模
        u_din = get_att(self.hist_it, item_emb_all, embedding_size, i_emb, self.sl)

        u_follow = get_att(self.hist_follow, item_emb_all, embedding_size, i_emb, self.sl_follow)
        u_vote = get_att(self.hist_vote, item_emb_all, embedding_size, i_emb, self.sl_vote)

        # 某一批次中某些行的sl为0 ，如何避免这种情况呢？
        #弱化u_follow和u_vote的表达，强化部分表达
        u_din = tf.layers.batch_normalization(inputs=u_din)
        u_follow = tf.layers.batch_normalization(inputs=u_follow)
        u_vote = tf.layers.batch_normalization(inputs=u_vote)

        uu_final = level_att(u_din, u_follow, u_vote, i_emb, embedding_size)
        inp = tf.concat([uu_final, i_emb], axis = -1)
        # weight_f = tf.expand_dims(self.weight_f, axis=1)
        # weight_v = tf.expand_dims(self.weight_v, axis=1)
        # inp = tf.concat([u_din, (1+weight_f)*u_follow, (1+weight_v)*u_vote, i_emb], axis=-1)

        dnn1 = tf.layers.dense(inp, 80,
                               kernel_initializer=layers.xavier_initializer(),
                               bias_initializer=tf.constant_initializer(0.01),
                               activation=tf.nn.tanh)
        dnn1 = tf.nn.dropout(dnn1, self.keep_prob)
        dnn2 = tf.layers.dense(dnn1, 1,
                               kernel_initializer=layers.xavier_initializer(),
                               bias_initializer=tf.constant_initializer(0.01),
                               activation=None)

        y_hat = tf.reshape(dnn2, [-1])
        self.logits = y_hat

        weight_f = tf.expand_dims(self.weight_f, axis=1)
        weight_v = tf.expand_dims(self.weight_v, axis=1)
        weight = (weight_f + weight_v) / 2
        print("weight", weight_v.shape)

        loss_y = tf.reduce_mean(
            (1+weight) * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        # loss_y = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=self.logits,
        #         labels=self.y)
        # )

        self.loss = loss_y
        # self.loss = loss_y + 0.1 * loss_1 + 0.1 * loss_2
        tf.summary.scalar('loss_y', loss_y)
        tf.summary.scalar('loss', self.loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)




    def train(self,sess, uij, lr, keep_prob):
        y_pred, loss,_ = sess.run([self.logits, self.loss, self.train_op],feed_dict={
            self.u : uij[0],
            self.i : uij[1],
            self.y : uij[2],

            # self.item_matrix : item_matrix,

            self.hist_it: uij[3],
            self.hist_follow: uij[4],
            self.hist_vote: uij[5],

            self.sl: uij[6],
            self.sl_follow: uij[7],
            self.sl_vote: uij[8],
            self.weight_f: uij[9],
            self.weight_v: uij[10],

            self.lr : lr,
            self.keep_prob : keep_prob,
            self.is_training: True
        })

        return y_pred, loss


    def eval(self, sess, uij, lr, keep_prob):
        res1 = sess.run(self.logits, feed_dict={
            self.u: uij[0],
            self.i: uij[1],

            # self.item_matrix: item_matrix,

            self.hist_it: uij[3],
            self.hist_follow: uij[4],
            self.hist_vote: uij[5],

            self.sl: uij[6],
            self.sl_follow: uij[7],
            self.sl_vote: uij[8],
            self.weight_f: uij[9],
            self.weight_v: uij[10],

            self.lr: lr,
            self.keep_prob: keep_prob,
            self.is_training: False
        })
        res2 = sess.run(self.logits, feed_dict={
            self.u: uij[0],
            self.i: uij[2],

            # self.item_matrix: item_matrix,

            self.hist_it: uij[3],
            self.hist_follow: uij[4],
            self.hist_vote: uij[5],

            self.sl: uij[6],
            self.sl_follow: uij[7],
            self.sl_vote: uij[8],
            self.weight_f: uij[9],
            self.weight_v: uij[10],

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


def get_att(hist_i, item_emb_all, embedding_size, i_emb, sl):
    max_item_length = tf.shape(hist_i)[1]
    h_emb = tf.nn.embedding_lookup(item_emb_all, hist_i)  # [B, T*100]
    print(max_item_length)
    h_emb = tf.reshape(h_emb, [-1, max_item_length, embedding_size])  # 【B, T, H】
    hist = attention(i_emb, h_emb, sl)  ##【B, H】  【B, T, H】  【B,】
    print("DIN_part之后:", hist.shape)
    # hist = tf.layers.batch_normalization(inputs=hist)
    hist = tf.reshape(hist, [-1, embedding_size])  # [B,H]
    return hist

def get_att_2(hist_i, item_emb_all, embedding_size, i_emb, sl):
    max_item_length = tf.shape(hist_i)[1]
    h_emb = tf.nn.embedding_lookup(item_emb_all, hist_i)  # [B, T*100]
    print(max_item_length)
    h_emb = tf.reshape(h_emb, [-1, max_item_length, embedding_size])  # 【B, T, H】
    hist = attention_2(i_emb, h_emb, sl)  ##【B, H】  【B, T, H】  【B,】
    print("DIN_part之后:", hist.shape)
    # hist = tf.layers.batch_normalization(inputs=hist)
    hist = tf.reshape(hist, [-1, embedding_size])  # [B,H]
    return hist



def attention(queries, keys, keys_length):

    queries = tf.expand_dims(queries, axis=1)
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs) # B * 1 * T

    # Weighted Sum
    outputs = tf.matmul(outputs,keys) # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs


def attention_2(queries, keys, keys_length):

    queries = tf.expand_dims(queries, axis=1)
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs) # B * 1 * T

    pandding = tf.ones_like(outputs) * (0)
    outputs = tf.where(outputs > 0.05, outputs, pandding)

    # Weighted Sum
    outputs = tf.matmul(outputs, keys) # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs

def level_att(u_din, u_follow, u_vote, i_emb, embedding_size):
    inp = tf.concat([u_din, u_follow, u_vote], axis=-1)
    keys = tf.reshape(inp, [-1, 3, embedding_size])  #B*3*H

    queries = tf.expand_dims(i_emb, axis=1)    #B*1*H
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # B * 1 * T

    pandding = tf.ones_like(outputs) * (0)
    outputs = tf.where(outputs > 0.3, outputs, pandding)

    # Weighted Sum
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
    outputs = tf.reshape(outputs, [-1, embedding_size])
    return outputs

