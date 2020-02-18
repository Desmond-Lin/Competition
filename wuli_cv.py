import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from mpl_toolkits.mplot3d import Axes3D

t = time.time()
train_df_source = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//train.csv')
test_df_source = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//test.csv')
event_df_source = pd.read_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//event.csv')

# flag 0: 0.370793 1: 0.629207
# test = df.groupby('flag').count()
# test = test['x'].apply(lambda x: x/test['x'].sum())

# 噪声数量比较稳定，真实信号数量随event_id变化较大
# test = df.groupby(['flag','event_id']).count()['x']
# test = df[df['flag']==0].groupby('event_id').count()['x']
# plt.plot(test)
# plt.show()


# # 把t加上合成三维坐标图，点的大小用q表示
# df_event = train_df_source[train_df_source['event_id']==18]
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(df_event[df_event['flag']==1]['x'], df_event[df_event['flag']==1]['y'], df_event[df_event['flag']==1]['t'], c='r', s= df_event[df_event['flag']==1]['q'], label='signal')
# ax.scatter(df_event[df_event['flag']==0]['x'], df_event[df_event['flag']==0]['y'], df_event[df_event['flag']==0]['t'], c='g',s= df_event[df_event['flag']==0]['q'], label='noise')
# ax.legend(loc='best')
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# plt.show()

print(
    '============================================== feature engineering ==============================================')


def f_eng(df, event_df, is_train=True):
    df = pd.merge(df, event_df.loc[:, ['event_id', 'energymc', 'thetamc', 'phimc', 'xcmc', 'ycmc']], how='left',
                  on='event_id')
    # df['dis'] = np.sqrt(df['x']**2+df['y']**2+df['t']**2)
    # 时间差
    df['t_o'] = df['t'] - df['terror']
    df['q_mean'] = df['q'] - df['q'].groupby(df['event_id']).transform(np.mean)
    df['t_mean'] = df['t'] - df['t'].groupby(df['event_id']).transform(np.mean)
    if is_train:
        df = df.drop('flag', axis=1)
    df = df.drop(['hit_id', 'z', 'event_id'], axis=1)
    return df


labels = train_df_source['flag']
train_df = f_eng(train_df_source, event_df_source)
test_df = f_eng(test_df_source, event_df_source, False)
print(train_df.columns.values.tolist())
# 小样本训练
train_df = train_df.iloc[:100000, :]
labels = labels[:100000]

threshold = 0.5


# 自定义Metric函数
def score(labels, pred):
    # labels = train_data.get_label()
    labels = labels.astype(int)

    pred = np.array([1 if x >= threshold else 0 for x in pred])
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

    w1 = 99
    w2 = 1
    metric = w1 * remaining + w2 * exemption
    return metric


n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
eval_fun = f1_score

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'num_leaves': 15,
    'max_depth': 8,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'binary',
    'n_estimators': 5000,
    'metric': 'auc',
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}


def run_oof(clf, X_train, y_train, X_test, kf):
    print(clf)
    preds_train = np.zeros((len(X_train), 2), dtype=np.float)
    preds_test = np.zeros((len(X_test), 2), dtype=np.float)
    train_loss = []
    eval_loss = []

    i = 1
    for train_index, eval_index in kf.split(X_train, y_train):
        x_tr = X_train.loc[train_index]
        x_eval = X_train.loc[eval_index]
        y_tr = y_train.loc[train_index]
        y_eval = y_train.loc[eval_index]
        clf.fit(x_tr, y_tr, eval_set=[(x_eval, y_eval)], early_stopping_rounds=200, verbose=True)

        train_loss.append(eval_fun(y_tr, np.argmax(clf.predict_proba(x_tr)[:], 1), average='macro'))
        eval_loss.append(eval_fun(y_eval, np.argmax(clf.predict_proba(x_eval)[:], 1), average='macro'))

        preds_train[eval_index] = clf.predict_proba(x_eval)[:]
        preds_test += clf.predict_proba(X_test)[:]

        print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], eval_loss[-1], np.mean(eval_loss)))
        print('-' * 50)
        i += 1
    print('Train: ', train_loss)
    print('Val: ', eval_loss)
    print('-' * 50)
    print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(eval_loss)))
    preds_test /= n_fold
    return preds_train, preds_test, np.mean(eval_loss)


train_pred, test_pred, eval_loss = run_oof(lgb.LGBMClassifier(**params), train_df, labels, test_df, skf)
test_df_source['flag_pred'] = np.argmax(test_pred, 1)
len(test_pred)
len(test_df_source)
sub = pd.concat([test_df_source['hit_id'], test_df_source['flag_pred'], test_df_source['event_id']], axis=1)
sub.to_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//output//sub_{}.csv'.format(eval_loss),
           index=False)
print('runtime:', time.time() - t)
print('finish.')
print('========================================================================================================')
