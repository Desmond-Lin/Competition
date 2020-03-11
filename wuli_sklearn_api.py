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

t = time.time()
logging.basicConfig(level=logging.INFO, filename='wuli.log',
                    filemode='a', format='%(asctime)s - %(name)s - %('
                                         'levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.info('---------------New Training Start------------------')
train = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//train.csv')
test = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//test.csv')
df = pd.concat([train, test],sort=True)
event = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//event.csv')
df = pd.merge(df, event, on='event_id', how='left')

del train, test, event
gc.collect()

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

# 自定义Metric函数
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

def get_stati_feature(df, col):
    for func in ['mean', 'quantile', 'skew', 'median', 'std']:
        dic = df[col].groupby(df['event_id']).agg(func).to_dict()
        col_name = col + '_' + func
        df[col_name] = df['event_id'].map(dic).values


def feature_engineer(df):
    logging.info('--------------  ----------------------')

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

    # 乱来的速度
    df['v'] = df['dis'] / df['t']
    df['v_'] = df['dis_'] / df['t']

    # x_minus_xcmc,y_minus_ycmc,t都分布在0附近高概率，直接求距离
    # df['pro_dis'] = np.sqrt(df['x_minus_xcmc'] ** 2 + df['y_minus_ycmc'] ** 2+df['t'] ** 2)

    # 按照公式，拟合簇射前峰面
    # l = np.sin(df['thetamc']) * np.cos(df['phimc'])
    # m = np.sin(df['thetamc']) * np.sin(df['phimc'])
    df['surface_fitting'] = df['t'] - (np.sin(df['thetamc']) * np.cos(df['phimc']) * df['x'] +  np.sin(df['thetamc']) * np.sin(df['phimc'])* df['y']) / 0.2998

    # 原初粒子的能量和次级粒子电量的比例关系.q为负电荷全部都是信号，不知道什么原因
    df['q_divide_energymc'] = df['q'] / df['energymc']
    # df['sumq_divide_energymc'] = df['q'].groupby(df['event_id']).transform(np.sum) / df['energymc']
    # df['q_minus_mean_e'] = df['q'] - df['energymc'] / df['nhitreal']

    df['q2'] = df['q'] ** 2

    df['slope'] = df['y'] / df['x']

    df['x2'] = df['x'] ** 2

    df['y2'] = df['y'] ** 2
    #
    # df['x_cmc'] = (df['x'] - df['xcmc']) / (df['xcmc'] + df['xcmc'].mean())

    # df['y_cmc'] = (df['y'] - df['ycmc']) / (df['ycmc'] + df['ycmc'].mean())

    df['nhitreal_percent'] = df['nhitreal'] / df['nhit']

    # t 统计特征
    df['event_id_t_min'] = df.groupby('event_id')['t'].transform('min')
    df['event_id_t_max'] = df.groupby('event_id')['t'].transform('max')
    df['event_id_t_median'] = df.groupby('event_id')['t'].transform('median')
    df['event_id_t_mean'] = df.groupby('event_id')['t'].transform('mean')

    # t "偏移"时间
    # df['t_min_diff'] = df['t'] - df['event_id_t_min']
    # df['t_max_diff'] = df['event_id_t_max'] - df['t']
    # df['t_median_diff'] = df['event_id_t_median'] - df['t']
    df['t_mean_diff'] = df['t']-df['event_id_t_mean']

    # # 统计特征
    get_stati_feature(df, 'x')
    get_stati_feature(df, 'y')
    get_stati_feature(df, 't')
    get_stati_feature(df, 'q')

    reduce_mem(df)
    # 重新排序, 为后面 t 的 rolling 或者 diff 特征做准备
    # df = df.sort_values(by=['event_id', 't_min_diff']).reset_index(drop=True)
    #
    # # 时间变化特征, 强特
    # # 也可以用 rolling 加不同 window_size .std() 来做, 效果比 diff 稍微差一点
    # # 试过 rolling + diff 效果比只用一种要差, 我还没搞清楚, 可以多尝试
    #
    # for i in [4, 6, 8, 10, 12]:
    #     df[f't_diff_last_{i}'] = df.groupby('event_id')['t'].diff(periods=i).fillna(0)
    #
    # # 位置的变化特征, 线上 +3 左右
    #
    # for i in range(1, 21):
    #     df[f'x_diff_last_{i}'] = df.groupby(['event_id'])['x'].diff(periods=i).fillna(0)
    #     df[f'y_diff_last_{i}'] = df.groupby(['event_id'])['y'].diff(periods=i).fillna(0)
    #
    # # 与中心距离的位置变化特征, 线上 +1 左右
    # df['dis2c'] = ((df['x'] - df['xcmc']) ** 2 + (df['y'] - df['ycmc']) ** 2) ** 0.5
    # for i in range(1, 10):
    #     df[f'dis2c_diff_last_{i}'] = df.groupby(['event_id'])['dis2c'].diff(periods=i).fillna(0)



    # 调下精度
    # reduce_mem(df)

    # 分类特征
    # df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')

    # 二阶特征组合
    # feature = ['x', 'y', 't', 'dis', 'x_minus_xcmc', 'y_minus_ycmc', 't_mean_diff', 'nhitreal_percent', 'thetamc',
    #            'phimc']
    feature = [ 't', 'x_minus_xcmc', 'y_minus_ycmc', 't_mean_diff', 'nhitreal_percent', 'thetamc',
               'phimc']
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
    df['x_y_tmean'] = df['x']/100 * df['y']/100 * df['t_mean_diff']

    df['xcmc_ycmc_tmean'] = df['x_minus_xcmc']/100 * df['y_minus_ycmc']/100 * df['t_mean_diff']

    return df


# # 伪标签
# df.loc[(df['is_train']==0)&(df['t']<-900),'flag']=0
# df.loc[(df['is_train']==0)&(df['t']<-900),'is_train']=1
# df.loc[(df['is_train']==0)&(df['t']>1850),'flag']=0
# df.loc[(df['is_train']==0)&(df['t']>1850),'is_train']=1
# df.loc[(df['is_train']==0)&(df['q']<0),'flag']=0
# df.loc[(df['is_train']==0)&(df['q']<0),'is_train']=1
# labels = df.loc[df['flag'].notnull(),'flag']

# feature engineering
df = feature_engineer(df)

id_and_label = ['event_id', 'hit_id', 'flag']
useless_features = [
    'z',
    'event_id_t_mean', 'event_id_t_median', 'event_id_t_min', 'event_id_t_max',
]
use_features = [col for col in df.columns if col not in id_and_label + useless_features]

# 类型特征
# cate_cols = ['x_y']
# for f in cate_cols:
#     map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
#     df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
# del map_dict
# gc.collect()

# df_iso = df.loc[:, ['x', 'y', 't', 'q', 't_minus_mean', 'thetamc','phimc', 'v','x_x_minus_xcmc','t_minus_mean_t_minus_mean','dis','q_divide_energymc']]
# reduce_mem(df)

# print(
#     '============================================== isolation forest ==============================================')
# # isolation forest先预测一下，预测结果作为一个特征输入
# rng = np.random.RandomState(42)
# clf = IsolationForest(behaviour='new', max_samples=100,
#                       random_state=rng, contamination='auto')
# clf.fit(df_iso)
# df['iso_pre'] = clf.decision_function(df_iso)
# del df_iso

labels = df.loc[df['flag'].notnull(),'flag']
train_df = df[use_features][df.flag.notna()]
test_df = df[use_features][df.flag.isna()]
sub = df[['hit_id', 'event_id']][df.flag.isna()]
del df
gc.collect()

print(train_df.columns.values.tolist())
logging.info(train_df.columns.values.tolist())
# 小样本训练
# train_df = train_df.iloc[:500000, :]
# labels = labels[:500000]

train_x, val_x, train_y, val_y = train_test_split(train_df, labels, test_size=0.33, random_state=42)

print('runtime:', time.time() - t)

print(
    '=============================================== training and validati ===============================================')
fea_imp_list = []
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=6000,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019,
    metric=None,
    n_jobs=20
)

print('************** training **************')
clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    # categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** validation result **************')
best_rounds = clf.best_iteration_
best_score = clf.best_score_['valid_0']['auc']
val_pred = clf.predict_proba(val_x)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print(
    '=============================================== whole dataset training  ===============================================')
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=best_rounds,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019,
    n_jobs=20
)

print('************** training **************')
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    eval_metric='auc',
    # categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** test result **************')
test_pre = pd.DataFrame(clf.predict_proba(test_df)[:, 1], columns=['flag_pred'])
print('test_pre runtime:', time.time() - t)
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print(
    '=============================================== feat importances ===============================================')
# 特征重要性可以好好看看
fea_imp_dict = dict(zip(train_df.columns.values, np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))
    logging.info('{} = {}'.format(f, imp))

print(
    '=============================================== threshold search ===============================================')
# f1阈值敏感，所以对阈值做一个简单的迭代搜索。
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
val_pred_copy = val_pred
for step in range(225):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in val_pred_copy]
    curr_f1 = score(val_y, y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('curr_t: {}   best threshold: {}   best score: {}'.format(curr_t, best_t, best_f1))
print('search finish.')

val_pred_copy = [1 if x >= best_t else 0 for x in val_pred_copy]
print('\nbest auc:', best_score)
print('best score:', score(val_y, val_pred_copy))
print('validate mean:', np.mean(val_pred_copy))
print('runtime:', time.time() - t)

print('=============================================== sub save ===============================================')
# sub.to_csv(
#     'C://Users//Lin//Desktop//PolyU//competition//turing_wuli//sub_prob_{}_{}_{}.csv'.format(best_score),
#     index=False)
sub = pd.concat([sub['hit_id'], test_pre, sub['event_id']], axis=1)
sub['flag_pred'] = sub['flag_pred'].apply(lambda x: 1 if x >= best_t else 0)
sub.to_csv(
    'C://Users//Lin//Desktop//PolyU//competition//turing_wuli//output//sub_{}_{}.csv'.format(best_t, best_score),
    index=False)
print('runtime:', time.time() - t)
logging.info('Best score: {}'.format(best_t))
print('flag == 1 count: {}'.format(sub['hit_id'][sub['flag_pred'] == 1].count()))
print('flag == 0 count: {}'.format(sub['hit_id'][sub['flag_pred'] == 0].count()))
print('finish.')
print('========================================================================================================')
