import numpy as np
import random
random.seed(1234)
np.random.seed(1234)
class DataInput:
  def __init__(self, train_data, batch_size):

    self.batch_size = batch_size
    self.train_data = train_data
    self.epoch_size = len(self.train_data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.train_data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size -1:
      raise StopIteration

    ts = self.train_data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.train_data))]
    # ts = ts[0:-1]
    self.i += 1

    u, i, y, sl, sl_follow, sl_vote = [], [], [], [], [], []
    list_it, list_follow, list_vote = [], [], []
    weig = []
#u, hist_it, hist_follow, hist_vote, target_id, 1
# (line[0], line[1], line[2], line[3], line[4], line[5], trans_fo_20, trans_vo_20,
#  mul_fo_20, mul_vo_20, trans_fo_cos, trans_vo_cos))
    for t in ts:   #对于每条训练样本
        u.append(t[0])   #user
        i.append(t[4])
        y.append(t[5])
        if t[7] >=0 and t[7]<=1:   #t[7] 为vote
            weig.append(t[7])
        else:
            weig.append(0)

        list_1 = t[1]
        if list_1 == []:
            list_1 = [0]
        list_it.append(list_1)
        sl.append(len(list_1))

        list_2 = t[2]
        if list_2 == []:
            list_2 = [0]
        list_follow.append(list_2)
        sl_follow.append(len(list_2))

        list_3 = t[3]
        if list_3 == []:
            list_3 = [0]
        list_vote.append(list_3)
        sl_vote.append(len(list_3))

    hist_it = zhuan_2D(sl, list_it)
    hist_follow = zhuan_2D(sl_follow, list_follow)
    hist_vote = zhuan_2D(sl_vote, list_vote)

    return self.i, (u, i, y, hist_it, hist_follow, hist_vote, sl, sl_follow, sl_vote, weig)


class DataInputTest:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size -1:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        # print(ts)
        # ts = ts[0:-1]
        self.i += 1

        u, i, j, sl, sl_follow, sl_vote = [], [], [], [], [], []
        list_it, list_follow, list_vote = [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[4][0])
            j.append(t[4][1])

            list_1 = t[1]
            if list_1 == []:
                list_1 = [0]
            list_it.append(list_1)
            sl.append(len(list_1))

            list_2 = t[2]
            if list_2 == []:
                list_2 = [0]
            list_follow.append(list_2)
            sl_follow.append(len(list_2))

            list_3 = t[3]
            if list_3 == []:
                list_3 = [0]
            list_vote.append(list_3)
            sl_vote.append(len(list_3))


        hist_it = zhuan_2D(sl, list_it)
        hist_follow = zhuan_2D(sl_follow, list_follow)
        hist_vote = zhuan_2D(sl_vote, list_vote)

        return self.i, (u, i, j, hist_it, hist_follow, hist_vote, sl, sl_follow, sl_vote)



def zhuan_2D(sl, list_it):
    max_sl = max(sl)  # 此batch内用户看过的max的长度
    # max_sl = 55
    hist_i = np.zeros([len(list_it), max_sl], np.int64)

    k = 0
    for t in list_it:  # 每一行训练集
        for l in range(max_sl-len(t), max_sl):  # 用户所看的每一个列表
            hist_i[k][l] = t[l-(max_sl-len(t))]
        k += 1
    return hist_i

