import pickle
import pandas as pd

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#
# weight_data_20 = pd.read_csv("t35/train_test_set_10_weight_200c_update.csv", header=None,
#                           names=['user_id', 'trans_fo_20','trans_vo_20',
#                                  'mul_fo_20','mul_vo_20'])
#
# df_fo_tr = pd.DataFrame(weight_data_20,columns = ['trans_fo_20'])
# df_fo_tr = df_fo_tr[(df_fo_tr['trans_fo_20'] <= 1) & (df_fo_tr['trans_fo_20'] >= 0)]
# df_vo_tr = pd.DataFrame(weight_data_20,columns = ['trans_vo_20'])
# df_vo_tr = df_vo_tr[(df_vo_tr['trans_vo_20'] <= 1) & (df_vo_tr['trans_vo_20'] >= 0)]
#
# df_mul_fo_tr = pd.DataFrame(weight_data_20,columns = ['mul_fo_20'])
# df_mul_fo_tr = df_mul_fo_tr[(df_mul_fo_tr['mul_fo_20'] <= 1) & (df_mul_fo_tr['mul_fo_20'] >= 0)]
# df_mul_vo_tr = pd.DataFrame(weight_data_20,columns = ['mul_vo_20'])
# df_mul_vo_tr = df_mul_vo_tr[(df_mul_vo_tr['mul_vo_20'] <= 1) & (df_mul_vo_tr['mul_vo_20'] >= 0)]
#
# print(float(df_fo_tr.var()), '\t', float(df_mul_fo_tr.var()), '\n',
#       float(df_vo_tr.var()), '\t', float(df_mul_vo_tr.var())) # 显示每一列的方差   0.144942    0.108092
# print(float(df_fo_tr.mean()), '\t', float(df_mul_fo_tr.mean()), '\n',
#       float(df_vo_tr.mean()), '\t', float(df_mul_vo_tr.mean()) )  #均值            0.107664     0.136637


# df_fo_mul = pd.DataFrame(weight_data_20,columns = ['mul_fo_20'])
# df_fo_mul = df_fo_mul[(df_fo_mul['mul_fo_20'] <= 1) & (df_fo_mul['mul_fo_20'] >= 0)]
#
# df_vo_mul = pd.DataFrame(weight_data_20,columns = ['mul_vo_20'])
# df_vo_mul = df_vo_mul[(df_vo_mul['mul_vo_20'] <= 1) & (df_vo_mul['mul_vo_20'] >= 0)]
#
# df_fo_cos = pd.DataFrame(weight_data_20,columns = ['cos_fo'])
# df_fo_cos = df_fo_cos[(df_fo_cos['cos_fo'] <= 1) & (df_fo_cos['cos_fo'] >= 0)]
#
# df_vo_cos = pd.DataFrame(weight_data_20,columns = ['cos_vo'])
# df_vo_cos = df_vo_cos[(df_vo_cos['cos_vo'] <= 1) & (df_vo_cos['cos_vo'] >= 0)]
# print(df_fo_tr)
# print(float(df_fo_tr.var()), '\t', float(df_fo_mul.var()), '\t', float(df_fo_cos.var()), '\n',
#       float(df_vo_tr.var()), '\t', float(df_vo_mul.var()), '\t', float(df_vo_cos.var())) # 显示每一列的方差   0.144942    0.108092
# print(float(df_fo_tr.mean()), '\t', float(df_fo_mul.mean()), '\t', float(df_fo_cos.mean()), '\n',
#       float(df_vo_tr.mean()), '\t', float(df_vo_mul.mean()), '\t', float(df_vo_cos.mean()))  #均值            0.107664     0.136637
# print(float(df_fo_tr.var()), '\n',
#       float(df_vo_tr.var())) # 显示每一列的方差   0.144942    0.108092
# print(float(df_fo_tr.mean()),'\n',
#       float(df_vo_tr.mean()) )  #均值            0.107664     0.136637

#weight_data_20 = pd.read_csv("train_test_set_5_cossim_update.csv", header=None,
#                           names=['user_id', 'cos_fo_20','cos_vo_20'])
# df_fo_tr = pd.DataFrame(weight_data_20,columns = ['cos_fo_20'])
# df_fo_tr = df_fo_tr[(df_fo_tr['cos_fo_20'] <= 1) & (df_fo_tr['cos_fo_20'] >= 0)]
# df_vo_tr = pd.DataFrame(weight_data_20,columns = ['cos_vo_20'])
# df_vo_tr = df_vo_tr[(df_vo_tr['cos_vo_20'] <= 1) & (df_vo_tr['cos_vo_20'] >= 0)]
#
# print(float(df_fo_tr.var()), '\n',
#       float(df_vo_tr.var())) # 显示每一列的方差   0.144942    0.108092
# print(float(df_fo_tr.mean()), '\n',
#       float(df_vo_tr.mean()))  #均值            0.107664     0.136637


weight_data_20 = pd.read_csv("train_test_set_5_oushidis.csv", header=None,
                          names=['user_id', 'cos_fo_20','cos_vo_20'])
df_fo_tr = pd.DataFrame(weight_data_20,columns = ['cos_fo_20'])
df_fo_tr = df_fo_tr[(df_fo_tr['cos_fo_20'] <= 30) & (df_fo_tr['cos_fo_20'] >= 0)]
df_vo_tr = pd.DataFrame(weight_data_20,columns = ['cos_vo_20'])
df_vo_tr = df_vo_tr[(df_vo_tr['cos_vo_20'] <= 30) & (df_vo_tr['cos_vo_20'] >= 0)]

print(float(df_fo_tr.var()), '\n',
      float(df_vo_tr.var())) # 显示每一列的方差   0.144942    0.108092
print(float(df_fo_tr.mean()), '\n',
      float(df_vo_tr.mean()))  #均值            0.107664     0.136637



# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# # plt.rc('font',family='Times New Roman')
# plt.rcParams.update({"font.size":16})
# # del matplotlib.font_manager.weight_dict['roman']
# # matplotlib.font_manager._rebuild()
#
# # result_data = weight_data
# df = pd.DataFrame(weight_data,columns = ['trans_fo_20','trans_vo_20',
#                                          'trans_fo_cos','trans_vo_cos'])
# df = df[(df['trans_fo_20'] <= 1) & (df['trans_fo_20'] >= 0)]
# df = df[(df['trans_vo_20'] <= 1) & (df['trans_vo_20'] >= 0)]
# df = df[(df['trans_fo_cos'] <= 1) & (df['trans_fo_cos'] >= 0)]
# df = df[(df['trans_vo_cos'] <= 1) & (df['trans_vo_cos'] >= 0)]
# df.plot.box()
# # plt.xlabel("category",fontsize=16)
# # plt.ylabel('The numbers of user behaviors',fontsize=16)
# plt.grid(linestyle="--", alpha=0.8)
# print(df.describe())#显示中位数、上下四分位数、标准偏差等内容
# plt.show()
