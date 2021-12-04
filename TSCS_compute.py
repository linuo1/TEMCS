import pickle
import pandas as pd

import numpy as np
import random
"""
主要目的是
1、统计训练集中每行样本的每个item的concept分布；
2、利用concept分布得到p(y|s,t),p(y|t)的大小【行为整体的是各个行为的并集】
3、计算不同样本的可迁移性大小；不同行为的可迁移性大小
"""

with open('train_test_set_5.pkl', 'rb') as f2:
    train_set = pickle.load(f2)
    test_set = pickle.load(f2)
    user_count, item_count = pickle.load(f2)
# print("user_count, item_count", user_count, item_count)

concept_data_20 = pd.read_csv("centers_30.csv")
concept_matrix_20 = concept_data_20.as_matrix()
concept_matrix_20 = concept_matrix_20.astype(np.float32)
concept_matrix_20 = concept_matrix_20[:,:100]
print("concept_matrix.shape", concept_matrix_20.shape) #30个点，每个点100维度

with open('item_matrix_real.pkl', 'rb') as f:
    item_matrix = pickle.load(f)
item_matrix = item_matrix.astype(np.float32)

def Jaccart_sim(x,y):
    p = np.sum(x & y)    #x,y均为 1 的个数
    q = np.sum(x | y)    #x、y中其中一个为 1 的个数
    return p/q

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


def get_seq00(target, concept_matrix):
    res_b = []
    for c in concept_matrix:
        rea_1 = np.linalg.norm(target - c)
        res_b.append(rea_1)
        # if rea_1 < 10:
        #     res_b.append(1)
        # else:
        #     res_b.append(0)
    res_b = np.array(res_b)
    ranges = res_b.max() - res_b.min()  # 最大值列表 - 最小值列表 = 差值列表
    b_c = res_b - res_b.min()
    b_final = b_c / ranges
    b_final = np.int64(b_final < 0.1)
    return b_final



def get_seq01(arr, concept_matrix):
    """
    输入一个数组arr和一个target t的表示，计算出最终的[0,1,1,0...]代码
    ***由于考虑的是有无促进作用，因为只要方向一致，便有促进作用，选择用余弦相似度计算,阈值为0
    #余弦错误，这是聚类的时候，所以是欧氏距离,但是阈值难以确定。。。假设每个点可以有5个concept，或者距离为10一下的。这是个参数哈
    :param arr:
    :return: [0,1,1,1,0]
    """

    res = []
    for behavior in arr:  #按理是21个行为，每个行为分别同30个点（即30个concept）同计算
        res_b = []
        for c in concept_matrix:
            rea_1 = np.linalg.norm(behavior - c)
            res_b.append(rea_1)
        res_b = np.array(res_b)
        ranges = res_b.max() - res_b.min()  # 最大值列表 - 最小值列表 = 差值列表
        b_c = res_b - res_b.min()
        b_final = b_c / ranges
        b_final = np.int64(b_final < 0.1)    #欧氏距离最小的几个
        res.append(b_final)

    # print("res", res)  # 21个30维度，得到1个30维度的内容，利用投票的原则，对于每个concept投票
    res = np.array(res)
    res_ult = res.sum(axis=0)  # 计算和，并求行数res.shape[0]/2
    # print("res_ult", res_ult)
    res_ult = np.int64(res_ult > res.shape[0] / 2)
    # print("final res_ult", res_ult)  # 30个concept [1 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]

    return res_ult


def get_sim_3(a,b,c):
    # res_3 = ((a==b) + 0) +((a==c) + 0)
    a = (a == 1)
    b = (b == 1)
    c = (c == 1)
    res = (a + 0) + (b + 0) + (c + 0)
    count = 0
    for i in res:
        if i == 3:
            count += 1
    if count == 0:
        return 1
    else:
        return count


def get_sim_2(a,b):
    # res_2 = ((a==b) + 0)
    a = (a == 1)
    b = (b == 1)
    res = (a + 0) + (b + 0)
    count = 0
    for i in res:
        if i == 2:
            count += 1
    if count == 0:
        return 1
    else:
        return count

def get_sim_ta(s1, target):
    count = get_sim_2(s1, target)
    if np.sum(target) == 0:
        p = 0.01
    else:
        p = count/np.sum(target)
    return p


def get_trans(sem_answer,sem_follow, sem_vote, sem_fo, sem_vo, concept_matrix_20):
    res_t = get_seq01(sem_answer, concept_matrix_20) #1个  30维度的内容
    res_sf = get_seq01(sem_follow, concept_matrix_20)   #follow + answer
    res_sv = get_seq01(sem_vote, concept_matrix_20)
    # res_yall = get_seq01(sem_yall, concept_matrix_20)   #answer all
    #
    res_f = get_seq01(sem_fo, concept_matrix_20)  #纯粹的follow
    res_v = get_seq01(sem_vo, concept_matrix_20)

    res_y = get_seq00(item_matrix[target], concept_matrix_20)
    print("1的个数", sum(res_y))

    print("sim", Jaccart_sim(res_sf, res_y), Jaccart_sim(res_t, res_y))
    trans_fo, trans_vo = 0.0, 0.0
    if (Jaccart_sim(res_t, res_y) == 1.0) :
        trans_fo = 0.0
        trans_vo = 0.0
    else:
        if (Jaccart_sim(res_sf, res_y) == 0.0):
            trans_fo = 0.0
        elif(Jaccart_sim(res_sv, res_y) == 0.0):
            trans_vo = 0.0
        elif (Jaccart_sim(res_t, res_y) == 0.0):
            trans_fo = np.log2(Jaccart_sim(res_sf, res_y)) / np.log2(Jaccart_sim(res_t, res_y) + 0.0001)
            trans_vo = np.log2(Jaccart_sim(res_sv, res_y)) / np.log2(Jaccart_sim(res_t, res_y) + 0.0001)
        else:
            trans_fo = np.log2(Jaccart_sim(res_sf, res_y)) / np.log2(Jaccart_sim(res_t, res_y))
            trans_vo = np.log2(Jaccart_sim(res_sv, res_y)) / np.log2(Jaccart_sim(res_t, res_y))

        # elif (Jaccart_sim(res_t, res_y) == 0.0):
        #     trans_fo = 1 - Jaccart_sim(res_f, res_yall) * np.log2(Jaccart_sim(res_sf, res_y)) / np.log2(Jaccart_sim(res_t, res_y) + 0.0001)
        #     trans_vo = 1 - Jaccart_sim(res_v, res_yall) * np.log2(Jaccart_sim(res_sv, res_y)) / np.log2(Jaccart_sim(res_t, res_y) + 0.0001)
        # else:
        #     trans_fo = 1 - Jaccart_sim(res_f, res_yall) * np.log2(Jaccart_sim(res_sf, res_y))/np.log2(Jaccart_sim(res_t, res_y))
        #     trans_vo = 1 - Jaccart_sim(res_v, res_yall) * np.log2(Jaccart_sim(res_sv, res_y)) / np.log2(Jaccart_sim(res_t, res_y))
    mul_fo, mul_vo = 0.0, 0.0
    if (Jaccart_sim(res_f, res_y) == 0.0):
        mul_fo = 0.0
    elif (Jaccart_sim(res_v, res_y) == 0.0):
        mul_vo = 0.0
    elif (sum(res_y) == concept_matrix_20.shape[0]):
        mul_fo = np.log2(Jaccart_sim(res_f, res_y))
        mul_vo = np.log2(Jaccart_sim(res_v, res_y))
    else:
        mul_fo = np.log2(Jaccart_sim(res_f, res_y)) / np.log2(sum(res_y)/concept_matrix_20.shape[0])
        mul_vo = np.log2(Jaccart_sim(res_v, res_y)) / np.log2(sum(res_y)/concept_matrix_20.shape[0])
    return trans_fo, trans_vo, mul_fo, mul_vo


##u, hist_it, hist_follow, hist_vote, target_id, 1
cou = 0
train_set_new = []
hang = []
user_tr_fo = {}
user_tr_vo = {}
user_mul_fo = {}
user_mul_vo = {}
for line in train_set:
    cou += 1
    user_id = line[0]

    list_answer = line[1]
    list_follow = line[2]
    list_vote = line[3]
    target = line[4]
    label = line[5]

    sem_answer = item_matrix[list_answer]
    sem_follow = item_matrix[list_follow + list_answer]
    sem_vote = item_matrix[list_vote + list_answer]

    sem_fo = item_matrix[list_follow]  # follow序列
    sem_vo = item_matrix[list_vote]

    trans_fo_20, trans_vo_20, mul_fo_20, mul_vo_20 = get_trans(sem_answer, sem_follow, sem_vote, sem_fo, sem_vo, concept_matrix_20)
    print("trans", trans_fo_20, trans_vo_20)

    if user_id in user_tr_fo.keys():
        user_tr_fo[user_id].append(trans_fo_20)
        user_tr_vo[user_id].append(trans_vo_20)
        user_mul_fo[user_id].append(mul_fo_20)
        user_mul_vo[user_id].append(mul_vo_20)
    else:
        user_tr_fo[user_id] = [trans_fo_20]
        user_tr_vo[user_id] = [trans_vo_20]
        user_mul_fo[user_id] = [mul_fo_20]
        user_mul_vo[user_id] = [mul_vo_20]


    # if cou > 20:
    #     break

#求字典数据的总和
user_tr_fo_final = {}
user_tr_vo_final = {}
user_mul_fo_final = {}
user_mul_vo_final = {}
for user in user_tr_fo.keys():
    data_list = user_tr_fo[user]
    sum_1 = 0
    for i in range(0, len(data_list)):
        sum_1 += data_list[i]
    user_tr_fo_final[user] = 1 - sum_1/len(data_list)
    print("user, user_tr_fo_final", user, user_tr_fo[user], user_tr_fo_final[user])

    data_list_vote = user_tr_vo[user]
    sum_vote = 0
    for j in range(0, len(data_list_vote)):
        sum_vote += data_list_vote[j]
    user_tr_vo_final[user] = 1 - sum_vote/len(data_list_vote)
    print("user, user_tr_vo_final", user, user_tr_vo[user], user_tr_vo_final[user])

    data_list_mul_follow = user_mul_fo[user]
    sum_fo = 0
    for j in range(0, len(data_list_mul_follow)):
        sum_fo += data_list_mul_follow[j]
    user_mul_fo_final[user] = 1 - sum_fo / len(data_list_mul_follow)
    print("user, user_mul_fo_final", user, user_mul_fo[user], user_mul_fo_final[user])

    data_list_mul_vote = user_mul_vo[user]
    sum_vo = 0
    for j in range(0, len(data_list_mul_vote)):
        sum_vo += data_list_mul_vote[j]
    user_mul_vo_final[user] = 1 - sum_vo / len(data_list_mul_vote)
    print("user, user_mul_vo_final", user, user_mul_vo[user], user_mul_vo_final[user])

for line in train_set:
    # cou += 1
    # user_id = line[0]
    #
    # list_answer = line[1]
    # list_follow = line[2]
    # list_vote = line[3]
    # target = line[4]
    # label = line[5]
    # train_set_new.append((line[0], line[1], line[2], line[3], line[4], line[5], user_tr_fo_final[line[0]], user_tr_vo_final[line[0]]))
    hang.append([line[0], user_tr_fo_final[line[0]], user_tr_vo_final[line[0]],
                 user_mul_fo_final[line[0]], user_mul_vo_final[line[0]]])




# with open('train_test_set_5_weight_30c_update.pkl', 'wb') as f:
#     pickle.dump(train_set_new, f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
#     pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)

# 保存为csv文件 #user_ID, trans_fo, trans_vo, trans_fo_cos, trans_vo_cos
data1 = pd.DataFrame(hang)
data1.to_csv("train_test_set_5_weight_30c_update.csv", index=0, encoding="UTF-8")


