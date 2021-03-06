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

df = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//df.csv')
reduce_mem(df)

# df.to_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//df.csv')

test = df[df.flag.isna()]

df.loc[df.flag.isna()&(df.t<-900), 'flag'] = 0
df.loc[df.flag.isna()&((df.t>1850)|(df.q<0)), 'flag'] = 1

train = df[df.flag.notna()]
train['flag'] = train['flag'].astype('int')

del df
gc.collect()

reduce_mem(test)
print('runtime:', time.time() - t)


def run_lgb(df_train, df_test, use_features):
    target = 'flag'
    oof_pred = np.zeros((len(df_train),))
    y_pred = np.zeros((len(df_test),))
    fea_imp_list = []

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
                          verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits

        print("Features importance...")
        fea_imp_list = fea_imp_list.append(model.feature_importance('gain'))
        # feat_imp = pd.DataFrame({'feature': model.feature_name(),
        #                          'split': model.feature_importance('split'),
        #                          'gain': 100 * fea_imp_list / fea_imp_list.sum()}).sort_values('gain', ascending=False)




        used_time = (time.time() - start_time) / 3600
        print(f'used_time: {used_time:.2f} hours')

        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()

    return y_pred, oof_pred , fea_imp_list

print(
    '=============================================== training and validati ===============================================')
y_pred, oof_pred, fea_imp_list = run_lgb(train, test, use_features)

print('=============================================== feat importances ===============================================')
# 特征重要性可以好好看看
fea_imp_dict = dict(zip(use_features, np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))


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
submission.to_csv(f'C://Users//Lin//Desktop//PolyU//competition//turing_wuli//output///submission_lgb_{score}_threshold_{best_t}.csv', index=False)  # 线上 54.600440509
submission.flag_pred.value_counts()
print('runtime:', time.time() - t)
logging.info('Best score: {}'.format(best_t))
print('flag == 1 count: {}'.format(submission['hit_id'][submission['flag_pred'] == 1].count()))
print('flag == 0 count: {}'.format(submission['hit_id'][submission['flag_pred'] == 0].count()))
print('finish.')
print('========================================================================================================')
