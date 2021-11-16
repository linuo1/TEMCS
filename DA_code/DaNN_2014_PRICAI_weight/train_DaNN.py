import pickle
import pandas as pd
import pickle

import numpy as np
import tensorflow as tf
import random

from model_DaNN import Model
from input_multi import DataInput, DataInputTest
from evalution_multi import evaluate_model
import time
import sys
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

epochs = 3
batch_size = 500
test_batch_size = 100
keep_prob = 0.8

def _eval(sess, test_set, model):
    auc_sum = 0.0
    for _, uij in DataInputTest(test_set, test_batch_size):
        # print("test uij", uij[7])
        lo = model.eval(sess, uij, lr, 1.0) * len(uij[0])
        #print(lo)  #264.0\276.0\250.0
        auc_sum += lo
    test_auc = auc_sum / len(test_set)
    return test_auc


with open('../../data/zhihu_data/item_matrix_10_split.pkl', 'rb') as f:
    item_matrix = pickle.load(f)
item_matrix = item_matrix.astype(np.float32)


with open('../../data/zhihu_data/train_test_set_10_weight_200c_update.pkl', 'rb') as f2:
    train_set = pickle.load(f2)
    test_set = pickle.load(f2)
    user_count, item_count = pickle.load(f2)
print(user_count, item_count)

for iii in range(3):
    lr = 0.001
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # with tf.Session() as sess:
        model = Model(user_count, item_count, item_matrix, batch_size)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        initAUC = _eval(sess, test_set, model)
        print('Init AUC: %.4f' % initAUC)

        best_HR = 0.0
        last_improved = 0
        require_improvement = 600
        flag = False
        for epoch in range(epochs):
            t1 = time.time()
            random.shuffle(train_set)
            epoch_size = round(len(train_set) / batch_size)

            loss_sum = 0.0
            j = 0
            for _, uij in DataInput(train_set, batch_size):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                result, loss = model.train(sess, uij, lr, keep_prob)
                print("loss:", loss, time.time() - t1)
                loss_sum += loss
                j += 1
            logloss = loss_sum / j

            t2 = time.time()
            print("前t2-t1:", t2 - t1)
            # f111.write("train time:" + '\t' + str(t2 - t1) + '\n')

            test_auc = _eval(sess, test_set, model)
            print("test_auc", test_auc)
            HR, NDCG, recall, precision, loss_test = evaluate_model(sess, model, test_set,
                                                                    10)
            # HR_1, NDCG_1, recall_1, precision_1, loss_test_1 = evaluate_model(sess, model, test_set,
            #                                                                        1)
            # HR_30, NDCG_30, recall_30, precision_30, loss_test_30 = evaluate_model(sess, model, test_set,
            #                                                                        30)
            # HR_40, NDCG_40, recall_40, precision_40, loss_test_40 = evaluate_model(sess, model, test_set,
            #                                                                        40)
            # HR_50, NDCG_50, recall_50, precision_50, loss_test_50 = evaluate_model(sess, model, test_set,
            #                                                                        50)

            print('epoch %d   \t[%.1f s]: recall= %.4f\tprecision= %.4f\t[%.1f s]' % (
                epoch, t2 - t1, recall, precision, time.time() - t2))
            print("test_auc, HR, NDCG", test_auc, HR, NDCG)

            f11 = open('Top10_DaNN_zhihu_BC_10_weight_200c_update.txt', 'a')
            f11.write(str(epoch) + '\t' + str(HR) + '\t' + str(NDCG) + '\t'
                      + str(recall) + '\t' + str(precision)
                      + '\t' + str(logloss) + '\t' + str(test_auc) + '\t' + str(loss_test) + '\n')
            # f11_20 = open('Top1_DaNN_zhihu_AC_5_weight_200c_update.txt', 'a')
            # f11_20.write(str(epoch) + '\t' + str(HR_1) + '\t' + str(NDCG_1) + '\t'
            #              + str(recall_1) + '\t' + str(precision_1)
            #              + '\t' + str(logloss) + '\t' + str(test_auc) + '\t' + str(loss_test_1) + '\n')
            # f11_30 = open('DIN_5_Top30m.txt', 'a')
            # f11_30.write(str(epoch) + '\t' + str(HR_30) + '\t' + str(NDCG_30) + '\t'
            #              + str(recall_30) + '\t' + str(precision_30)
            #              + '\t' + str(logloss) + '\t' + str(test_auc) + '\t' + str(loss_test_30) + '\n')
            # f11_40 = open('DIN_5_Top40m.txt', 'a')
            # f11_40.write(str(epoch) + '\t' + str(HR_40) + '\t' + str(NDCG_40) + '\t'
            #              + str(recall_40) + '\t' + str(precision_40)
            #              + '\t' + str(logloss) + '\t' + str(test_auc) + '\t' + str(loss_test_40) + '\n')
            # f11_50 = open('DIN_5_Top50m.txt', 'a')
            # f11_50.write(str(epoch) + '\t' + str(HR_50) + '\t' + str(NDCG_50) + '\t'
            #              + str(recall_50) + '\t' + str(precision_50)
            #              + '\t' + str(logloss) + '\t' + str(test_auc) + '\t' + str(loss_test_50) + '\n')

            # 如果1000步以后还没提升，就中止训练。
            if epoch - last_improved > require_improvement:
                print("No optimization for ", require_improvement, " steps, auto-stop in the ", epoch, " step!")
                # 跳出这个轮次的循环
                flag = True
                break
            # 跳出所有训练轮次的循环
            if flag:
                break
    # lr = lr * 0.8
    iii += 1
