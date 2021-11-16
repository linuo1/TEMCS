import numpy as np
import random
import math
random.seed(1234)
np.random.seed(1234)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def getHitRatio(items, iid):
    if iid in items:
        return 1.0
    else:
        return 0.0

def getNDCG(items, iid):
    for i in range(len(items)):
        if items[i] == iid:
            return math.log(2) / math.log(i + 2)
    return 0.0


def evaluate_model(sess, model, test_set, topk):
    hits, ndcgs, test_loss = [], [], []
    hit_2 = 0
    recall = 0
    precision = 0

    for t in test_set:
        u = t[0]     #用户
        i = t[4][0]    #候选的item
        # j = t[4][1]
        unseen_list = t[5][:]  #99个候选items
        # print(len(unseen_list)) #list赋值一大坑  =等同于地址赋值
        unseen_list.append(i)
        # print(unseen_list)
        items = np.array(unseen_list)  #测试集的items
        # print(len(items))
        users = np.full(len(items), u, dtype=np.int64)  # 将用户拉长，足以匹配items
        # print("users, items", users, items)


        list_1 = t[1]
        if list_1 == []:
            list_1 = [0]
        hist_it = np.tile(list_1, (len(items), 1))
        sl = np.full(len(items), len(list_1))

        list_2 = t[2]
        if list_2 == []:
            list_2 = [0]
        hist_follow = np.tile(list_2, (len(items), 1))
        sl_follow = np.full(len(items), len(list_2))

        list_3 = t[3]
        if list_3 == []:
            list_3 = [0]
        hist_vote = np.tile(list_3, (len(items), 1))
        sl_vote = np.full(len(items), len(list_3))


        weig_f = np.full(len(items), 0)
        weig_v = np.full(len(items), 0)
        predictions = sess.run(model.logits, feed_dict={
            model.u: users,
            model.i: items,
            # model.item_matrix: item_matrix,

            model.hist_it: hist_it,
            model.hist_follow: hist_follow,
            model.hist_vote: hist_vote,

            model.sl: sl,
            model.sl_follow: sl_follow,
            model.sl_vote: sl_vote,
            model.weight_f: weig_f,
            model.weight_v: weig_v,

            model.keep_prob: 1.0,
            model.is_training: False

        })
        # print(predictions)
        sorted_idx = np.argsort(predictions)[::-1]
        # print(sorted_idx[0:topk])
        selected_items = items[sorted_idx[0:topk]]

        ndcg = getNDCG(selected_items, i)
        hit = getHitRatio(selected_items, i)
        hits.append(hit)
        ndcgs.append(ndcg)

        if i in selected_items:
            hit_2 += 1
        recall += 1
        precision += topk

        y = []
        # len(a)个1,100-len(a)个0
        for i in range(100):
            if i >= 99:
                y.append(1.0)
            else:
                y.append(0.0)
        y = np.array(y)
        y_pred = sigmoid(predictions)
        log_loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        test_loss.append(np.mean(log_loss))

    return np.array(hits).mean(), np.array(ndcgs).mean(), hit_2/(recall*1.0), hit_2/(precision*1.0), np.array(test_loss).mean()

