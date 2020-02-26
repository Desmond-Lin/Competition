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
import logging
from sklearn.ensemble import IsolationForest

t = time.time()
logging.basicConfig(level=logging.INFO, filename='wuli.log',
                    filemode='a', format='%(asctime)s - %(name)s - %('
                                         'levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.info('---------------New Training Start------------------')
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
    '============================================== feature analysis ==============================================')


def f_ana(df):
    analysis = []
    for col in df.columns.values.tolist():
        analysis.append(df[col].min())
        analysis.append(df[col].max())
        analysis.append(df[col].mean())
        analysis.append(df[col].quantile(0.25))
        analysis.append(df[col].quantile(0.5))
        analysis.append(df[col].quantile(0.75))
        analysis.append(df[col].std())
    analysis = pd.DataFrame(np.array(analysis).reshape(df.shape[1], int(len(analysis) / df.shape[1])).T)
    analysis.columns = df.columns.values.tolist()
    analysis['index'] = ['min', 'max', 'mean', '1/4', '1/2', '3/4', 'std']
    analysis.set_index(["index"], inplace=True)
    return analysis


tmp = f_ana(train_df_source)
print('runtime:', time.time() - t)

print(
    '============================================== feature engineering ==============================================')


def get_stati_feature(df, col):
    for func in ['mean', 'quantile', 'skew', 'median', 'max', 'min', 'std']:
        dic = df[col].groupby(df['event_id']).agg(func).to_dict()
        col_name = col + '_' + func
        df[col_name] = df['event_id'].map(dic).values


def f_eng(df, event_df, is_train=True):
    df = pd.merge(df, event_df.loc[:, ['event_id', 'energymc', 'thetamc', 'phimc', 'xcmc', 'ycmc']], how='left',
                  on='event_id')
    # 当前hit和当前event统计值作差
    df['t_minus_terror'] = df['t'] - df['terror']
    df['t_plus_terror'] = df['t'] + df['terror']
    df['q_minus_mean'] = df['q'] - df['q'].groupby(df['event_id']).transform(np.mean)
    df['t_minus_mean'] = df['t'] - df['t'].groupby(df['event_id']).transform(np.mean)

    # 当前hit和当前事件触发hit的距离
    df['x_minus_xcmc'] = df['x'] - df['xcmc']
    df['y_minus_ycmc'] = df['y'] - df['ycmc']
    df['dis'] = np.sqrt(df['x_minus_xcmc'] ** 2 + df['y_minus_ycmc'] ** 2 )
    df['dis_3dim'] = np.sqrt(df['x_minus_xcmc'] ** 2 + df['y_minus_ycmc'] ** 2 + (df['t'] * 29.98*100) ** 2)

    # 按照公式，拟合簇射前峰面
    l = np.sin(df['thetamc']) * np.cos(df['phimc'])
    m = np.sin(df['thetamc']) * np.sin(df['phimc'])
    df['surface_fitting'] = df['t'] - (l * df['x'] + m * df['y'])/0.2998

    # 原初粒子的能量和次级粒子电量的比例关系，min-max数据标准化放大差距.q为负电荷全部都是信号，不知道什么原因
    df['q_divide_energymc'] = df['q'] / df['energymc']
    df['q_divide_energymc'] = (df['q_divide_energymc'] - df['q_divide_energymc'].min()) / (
                df['q_divide_energymc'].max() - df['q_divide_energymc'].min())

    # 测试后留下来的统计特征x_std, x_mean, x_skew, y_std, y_mean, y_skew, t_std, t_skew, q_quantile, q_min, q_skew, q_mean
    # get_stati_feature(df,'x')
    feature = ['x_std', 'x_mean', 'x_skew', 'y_std', 'y_mean', 'y_skew', 't_std', 't_skew', 'q_quantile', 'q_min',
               'q_skew', 'q_mean']
    for fea in feature:
        col = fea.split('_')[0]
        func = fea.split('_')[1]
        dic = df[col].groupby(df['event_id']).agg(func).to_dict()
        df[fea] = df['event_id'].map(dic).values

    if is_train:
        df = df.drop('flag', axis=1)
    df = df.drop(['hit_id', 'z', 'event_id'], axis=1)

    # isolation forest先预测一下，预测结果作为一个特征输入
    # rng = np.random.RandomState(42)
    # clf = IsolationForest(behaviour='new', max_samples=100,
    #                       random_state=rng, contamination='auto')
    # clf.fit(df)
    # df['iso_pre'] = clf.decision_function(df)

    return df


labels = train_df_source['flag']
train_df = f_eng(train_df_source, event_df_source)
test_df = f_eng(test_df_source, event_df_source, False)
print(train_df.columns.values.tolist())
# 小样本训练
train_df = train_df.iloc[:100000, :]
labels = labels[:100000]

train_x, val_x, train_y, val_y = train_test_split(train_df, labels, test_size=0.33, random_state=42)

print('runtime:', time.time() - t)


# 自定义Metric函数
threshold = 0.5
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

    w1 = 0.9
    w2 = 0.1
    metric = w1 * remaining + w2 * exemption
    return 'score', metric, True


print(
    '=============================================== training and validati ===============================================')
fea_imp_list = []

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'num_leaves': 255,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'binary',
    'n_estimators': 20000,
    'metric': 'auc',
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
    }
gbm=None

try:
    with open("/content/drive/My Drive/lgb_model_v20.txt","r") as f:
      gbm = lgb.Booster({'model_str':f.read()})
except IOError:
    print("File is not accessible.")


train_dataset = lgb.Dataset(train_x,train_y)
val_dataset = lgb.Dataset(val_x, val_y, reference=train_dataset)
gbm = lgb.train(params,train_dataset, valid_sets =val_dataset, init_model = gbm, early_stopping_rounds=200,  keep_training_booster=True)
# gbm.save_model("/content/drive/My Drive/lgb_model.txt")
gbm.save_model("lgb_model.txt")
test_pre = pd.DataFrame(gbm.predict(test_df)[:, 1], columns='flag_pred')
fea_imp_list.append(gbm.feature_importances_)
print('test_pre runtime:', time.time() - t)
sub = pd.concat([test_df_source['hit_id'], test_pre,test_df_source['event_id']], axis=1)


print('runtime:', time.time() - t)

print('************** save result **************')
fea_imp_dict = dict(zip(train_df.columns.values, fea_imp_list))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))
    logging.info('{} = {}'.format(f, imp))

print('runtime:', time.time() - t)


print('=============================================== sub save ===============================================')
# sub.to_csv(
#     'C://Users//Lin//Desktop//PolyU//competition//turing_wuli//sub_prob_{}_{}_{}.csv'.format(best_score),
#     index=False)
sub['flag_pred'] = sub['flag_pred'].apply(lambda x: 1 if x >= threshold else 0)
sub.to_csv('C://Users//Lin//Desktop//PolyU//competition//turing_wuli//sub_{}.csv'.format(time.time()), index=False)

print('runtime:', time.time() - t)
print('finish.')
print('========================================================================================================')
