import warnings
warnings.simplefilter('ignore')
import itertools
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

import pandas as pd
import numpy as np
numpy.random.seed(7)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from mpl_toolkits.mplot3d import Axes3D
import logging
from sklearn.ensemble import IsolationForest
import gc
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

t = time.time()
logging.basicConfig(level=logging.INFO, filename='wuli_lgb.log',
                    filemode='a', format='%(asctime)s - %(name)s - %('
                                         'levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.info('---------------New Training Start------------------')


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def score(labels, pred):
    # labels = train_data.get_label()

    pred = np.array([1 if x >= 0.5 else 0 for x in pred])
    # print(labels.__class__)
    # print(pred.__class__)
    # print(labels)
    # print(pred)
    labels = labels.astype(int)
    # Nhitrealrange：真实为信号，预测也为信号的数量
    nhitrealrange = sum(labels & pred)

    # nhitrange：训练后被认定为信号的数量
    nhitrange = sum(pred)

    # Nhitreal：事例中的信号hit数量
    nhitreal = sum(labels)

    # Nhit: 事例中的hit数量
    nhit = len(labels)

    # 保留比例
    remaining = nhitrealrange / nhitreal

    # 噪声排除比
    exemption = 1 - (nhitrange - nhitrealrange) / (nhit - nhitreal)

    # accuracy
    # check = [1 if x==y else 0 for x, y in zip(pred, labels)]
    # print("accuracy: {}".format(sum(check)/len(check)))

    w1 = 77
    w2 = 23
    metric = w1 * remaining + w2 * exemption
    # return 'score', metric, True
    return metric


print(
    '============================================== feature engineering ==============================================')

# 导入数据

train = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//train.csv')
test = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//test.csv')
df = pd.concat([train, test])
event = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//event.csv')
df = pd.merge(df, event, on='event_id', how='left')

# t "偏移"时间
df['t_min_diff'] = df['t'] - df['event_id_t_min']

# 重新排序, 为后面 t 的 rolling 或者 diff 特征做准备
df = df.sort_values(by=['event_id', 't_min_diff']).reset_index(drop=True)

y_train = df.loc[df['flag'].notnull(),'flag']

id_and_label = ['event_id', 'hit_id', 'flag']
useless_features = [
    'z',
    'event_id_t_mean', 'event_id_t_median', 'event_id_t_min', 'event_id_t_max',
]
use_features = [col for col in df.columns if col not in id_and_label + useless_features]


x_train = df[use_features][df.flag.notna()]
test = df[use_features][df.flag.isna()]

