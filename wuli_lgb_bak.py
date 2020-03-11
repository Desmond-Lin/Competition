import warnings

warnings.simplefilter('ignore')
import gc
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools

import pandas as pd
import numpy as np
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
from tqdm import tqdm

tqdm.pandas()

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

del train, test, event
gc.collect()

# t 统计特征
df['event_id_t_min'] = df.groupby('event_id')['t'].transform('min')
df['event_id_t_max'] = df.groupby('event_id')['t'].transform('max')
df['event_id_t_median'] = df.groupby('event_id')['t'].transform('median')
df['event_id_t_mean'] = df.groupby('event_id')['t'].transform('mean')

# 当前hit和当前event统计值作差
df['t_minus_terror'] = df['t'] - df['terror']
df['t_plus_terror'] = df['t'] + df['terror']
df['q_minus_mean'] = df['q'] - df['q'].groupby(df['event_id']).transform(np.mean)
df['x_minus_mean'] = df['x'] - df['x'].groupby(df['event_id']).transform(np.mean)
df['y_minus_mean'] = df['y'] - df['y'].groupby(df['event_id']).transform(np.mean)

# 当前hit和当前事件触发hit的距离
df['x_minus_xcmc'] = df['x'] - df['xcmc']
df['y_minus_ycmc'] = df['y'] - df['ycmc']
df['dis'] = np.sqrt(df['x_minus_xcmc'] ** 2 + df['y_minus_ycmc'] ** 2)
df['dis_'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)

# x_minus_xcmc,y_minus_ycmc,t都分布在0附近高概率，直接求距离
df['pro_dis'] = np.sqrt(df['x_minus_xcmc'] ** 2 + df['y_minus_ycmc'] ** 2 + df['t'] ** 2)

# 乱来的速度
df['v'] = df['dis'] / df['t']
df['v_'] = df['dis_'] / df['t']

# 按照公式，拟合簇射前峰面
# l = np.sin(df['thetamc']) * np.cos(df['phimc'])
# m = np.sin(df['thetamc']) * np.sin(df['phimc'])
df['surface_fitting'] = df['t'] - (
        np.sin(df['thetamc']) * np.cos(df['phimc']) * df['x'] + np.sin(df['thetamc']) * np.sin(df['phimc']) * df[
    'y']) / 0.2998

# 原初粒子的能量和次级粒子电量的比例关系.q为负电荷全部都是信号，不知道什么原因
df['q_divide_energymc'] = df['q'] / df['energymc']
# df['sumq_divide_energymc'] = df['q'].groupby(df['event_id']).transform(np.sum) / df['energymc']
# df['q_minus_mean_e'] = df['q'] - df['energymc'] / df['nhitreal']

df['q2'] = df['q'] ** 2

df['slope'] = df['y'] / df['x']

# df['x_cmc'] = (df['x'] - df['xcmc']) / (df['xcmc'] + df['xcmc'].mean())

# df['y_cmc'] = (df['y'] - df['ycmc']) / (df['ycmc'] + df['ycmc'].mean())

df['nhitreal_percent'] = df['nhitreal'] / df['nhit']

# t "偏移"时间
df['t_min_diff'] = df['t'] - df['event_id_t_min']
df['t_max_diff'] = df['event_id_t_max'] - df['t']
df['t_median_diff'] = df['event_id_t_median'] - df['t']
df['t_mean_diff'] = df['event_id_t_mean'] - df['t']

# # 统计特征
# get_stati_feature(df, 'x')
# get_stati_feature(df, 'y')
# get_stati_feature(df, 't')
# get_stati_feature(df, 'q')

# 降低内存
reduce_mem(df)

# 重新排序, 为后面 t 的 rolling 或者 diff 特征做准备
df = df.sort_values(by=['event_id', 't_min_diff']).reset_index(drop=True)

# 时间变化特征, 强特
# 也可以用 rolling 加不同 window_size .std() 来做, 效果比 diff 稍微差一点
# 试过 rolling + diff 效果比只用一种要差, 我还没搞清楚, 可以多尝试

for i in [4, 6, 8, 10, 12]:
    df[f't_diff_last_{i}'] = df.groupby('event_id')['t'].diff(periods=i).fillna(0)

# 修正时间, 没太大作用

df['t_minus_terror'] = df['t'] - df['terror']

# 位置与中心位置的比例?

df['x_div_xcmc'] = df['x'] / (df['xcmc'] + 0.01)
df['y_div_ycmc'] = df['y'] / (df['ycmc'] + 0.01)

# 位置的变化特征, 线上 +3 左右
for i in range(1, 21):
    df[f'x_diff_last_{i}'] = df.groupby(['event_id'])['x'].diff(periods=i).fillna(0)
    df[f'y_diff_last_{i}'] = df.groupby(['event_id'])['y'].diff(periods=i).fillna(0)


df['x2'] = df['x'] ** 2
df['y2'] = df['y'] ** 2

# 分类特征
df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')

# 二阶特征组合
feature = ['x', 'y', 't', 'dis', 'x_minus_xcmc', 'y_minus_ycmc', 't_mean_diff', 'nhitreal_percent', 'thetamc', 'phimc']
ls = itertools.combinations(feature, 2)
for first_fea, second_fea in ls:
    col = first_fea + '*' + second_fea
    if df[first_fea].max() > 100000 or df[second_fea].max() > 100000:
        df[col] = df[first_fea] / 10000 * df[second_fea]
    if df[first_fea].max() > 10000 or df[second_fea].max() > 10000:
        df[col] = df[first_fea] / 100 * df[second_fea]
    else:
        df[col] = df[first_fea] * df[second_fea]

# 三阶特征组合
df['x_y_tmean'] = df['x'] / 100 * df['y'] / 100 * df['t_mean_diff']

df['xcmc_ycmc_tmean'] = df['x_minus_xcmc'] / 100 * df['y_minus_ycmc'] / 100 * df['t_mean_diff']

# 与中心距离的位置变化特征, 线上 +1 左右
df['dis2c'] = ((df['x'] - df['xcmc']) ** 2 + (df['y'] - df['ycmc']) ** 2) ** 0.5
for i in range(1, 10):
    df[f'dis2c_diff_last_{i}'] = df.groupby(['event_id'])['dis2c'].diff(periods=i).fillna(0)


# freq encoding
# 没太大作用, 线上 +0.1

def freq_enc(df, col):
    vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(vc)
    return df


df = freq_enc(df, 'terror')

# 类型特征
cate_cols = ['x_y']
for f in cate_cols:
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')

id_and_label = ['event_id', 'hit_id', 'flag']
useless_features = [
    'z',
    'event_id_t_mean', 'event_id_t_median', 'event_id_t_min', 'event_id_t_max',
    'nhit',
]
use_features = [col for col in df.columns if col not in id_and_label + useless_features]
logging.info(use_features)

# 伪标签, 还算有点用, 线上 +0.5
# 这个规则是观察 train 数据得出

# t < -900 ==> 0
# t > 1850 ==> 1
# q < 0    ==> 1


test = df[df.flag.isna()]

df.loc[df.flag.isna() & (df.t < -900), 'flag'] = 0
df.loc[df.flag.isna() & ((df.t > 1850) | (df.q < 0)), 'flag'] = 1

train = df[df.flag.notna()]
train['flag'] = train['flag'].astype('int')

del df
gc.collect()
print('runtime:', time.time() - t)


def run_lgb(df_train, df_test, use_features):
    target = 'flag'
    oof_pred = np.zeros((len(df_train),))
    y_pred = np.zeros((len(df_test),))
    imp_list = []

    folds = GroupKFold(n_splits=6)  # 6 折比 5 折好一点, 当然有时间有机器可以试下更多的 folds
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[target], train['event_id'])):
        start_time = time.time()
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'learning_rate': 0.2,
            'metric': 'auc',
            'objective': 'binary',
            'feature_fraction': 0.80,
            'bagging_fraction': 0.75,
            'bagging_freq': 2,
            'n_jobs': -1,
            'seed': 1029,
            'max_depth': 8,
            'num_leaves': 64,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5
        }

        model = lgb.train(params,
                          train_set,
                          num_boost_round=5000,
                          early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          categorical_feature=cate_cols,
                          verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits

        imp_list.append(model.feature_importance('gain'))
        # feat_imp = pd.DataFrame({'feature': model.feature_name(),
        #                          'split': model.feature_importance('split'),
        #                          'gain': 100 * fea_imp_list / fea_imp_list.sum()}).sort_values('gain', ascending=False)

        used_time = (time.time() - start_time) / 3600
        print(f'used_time: {used_time:.2f} hours')

        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()

    # 特征重要性
    fea_imp_dict = dict(zip(use_features, np.mean(imp_list, axis=0)))
    fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
    for f, imp in fea_imp_item:
        print('{} = {}'.format(f, imp))
        logging.info('{} = {}'.format(f, imp))

    return y_pred, oof_pred


print(
    '=============================================== training and validati '
    '===============================================')
y_pred, oof_pred = run_lgb(train, test, use_features)

print(
    '=============================================== feat importances ===============================================')


print(
    '=============================================== threshold search ===============================================')
# f1阈值敏感，所以对阈值做一个简单的迭代搜索。
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
for step in range(225):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in oof_pred]
    curr_f1 = score(train['flag'], y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('curr_t: {}   best threshold: {}   best score: {}'.format(curr_t, best_t, best_f1))
print('search finish.')

print('best score:', best_f1)
print('runtime:', time.time() - t)

score = roc_auc_score(train['flag'], oof_pred)
print('auc: ', score)

# np.save(f'lgb_y_pred_{score}', y_pred)
# np.save(f'lgb_oof_pred_{score}', oof_pred)

# best_threshold = 0.35

print('=============================================== sub save ===============================================')
test['flag_pred'] = y_pred
submission = test[['hit_id', 'flag_pred', 'event_id']]
submission['flag_pred'] = submission['flag_pred'].apply(lambda x: 1 if x > best_t else 0)
submission = submission.sort_values(by='hit_id')
submission.to_csv(
    f'C://Users//Lin//Desktop//PolyU//competition//turing_wuli//output///submission_lgb_{score}_threshold_{best_t}.csv',
    index=False)  # 线上 54.600440509
submission.flag_pred.value_counts()
print('runtime:', time.time() - t)
logging.info('Best score: {}'.format(best_t))
print('flag == 1 count: {}'.format(submission['hit_id'][submission['flag_pred'] == 1].count()))
print('flag == 0 count: {}'.format(submission['hit_id'][submission['flag_pred'] == 0].count()))
print('finish.')
print('========================================================================================================')
