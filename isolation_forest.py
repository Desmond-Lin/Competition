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

    # # isolation forest先预测一下，预测结果作为一个特征输入
    # rng = np.random.RandomState(42)
    # clf = IsolationForest(behaviour='new', max_samples=100,
    #                       random_state=rng, contamination='auto')
    # clf.fit(df)
    # df['iso_pre'] = clf.decision_function(df)

    return df


labels = train_df_source['flag']
train_df = f_eng(train_df_source, event_df_source)

test_df = f_eng(test_df_source, event_df_source, False)