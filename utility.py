import gc
import random
import chainer
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


def factorize(df):
    columns = df.select_dtypes([np.object]).columns.tolist()
    for c in columns:
        df[c], _ = pd.factorize(df[c])


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def split_train(df):
    pos_df = df[df['TARGET'] == 1].sample(frac=1)
    neg_df = df[df['TARGET'] == 0].sample(frac=1)
    n_pos = pos_df.shape[0]
    n_neg = neg_df.shape[0]
    n_pos_train = int(0.85*n_pos)
    n_neg_train = int(0.85*n_neg)
    train_df = pd.concat([pos_df[:n_pos_train], neg_df[:n_neg_train]])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([pos_df[n_pos_train:], neg_df[n_neg_train:]])
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    return train_df, test_df


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(
        df, columns=categorical_columns, dummy_na=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def calc_2nd_order_feature(df):
    columns = df.select_dtypes([np.number]).columns.tolist()
    for c in [
        'SK_ID_CURR',
        'SK_ID_PREV',
        'SK_ID_BUREAU',
    ]:
        try:
            columns.remove(c)
        except Exception as e:
            continue

    grp = df.groupby('SK_ID_CURR')[columns].sum().reset_index()
    tcolumns = []
    for i in range(len(columns)):
        a = columns[i]
        for j in range(i+1, len(columns)):
            b = columns[j]
            c = 'NUMERIC_DIV_{}__{}'.format(a, b)
            d = 'NUMERIC_MUL_{}__{}'.format(a, b)
            e = 'NUMERIC_ADD_{}__{}'.format(a, b)
            f = 'NUMERIC_MIN_{}__{}'.format(a, b)
            tcolumns += [c, d, e, f]
            grp[c] = grp[a] / grp[b]
            grp[d] = grp[a] * grp[b]
            grp[e] = grp[a] + grp[b]
            grp[f] = grp[a] - grp[b]

    grp = grp[['SK_ID_CURR'] + tcolumns]
    return grp


def reduce_memory(df):
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
        if df[c].dtype == 'int64':
            df[c] = df[c].astype('int32')


def filter_by_corr(df):
    train = pd.read_feather('./data/application_train.feather')
    train = train[['SK_ID_CURR', 'TARGET']]
    train = train.merge(df, on='SK_ID_CURR')
    train = train.drop(['SK_ID_CURR'], 1)
    corr = train.corr()[['TARGET']]
    corr['TARGET'] = corr['TARGET'].apply(abs)
    corr = corr.sort_values('TARGET', ascending=False)
    print(corr)

    cols = corr[corr['TARGET'] > 0.01].index.values.tolist()
    cols.remove('TARGET')
    if 'SK_ID_CURR' not in cols:
        cols.append('SK_ID_CURR')
    return df[cols]


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def filter_by_lgb(df, thres=3.00):
    print(df.shape)
    orig = df.copy()
    lgbc = LGBMClassifier(
        objective='binary', n_estimators=500, learning_rate=0.1, num_leaves=20,
        subsample=1.0, colsample_bytree=1.0, reg_alpha=100, reg_lambda=100, min_split_gain=0.01,
        min_child_weight=40, n_jobs=12,
        random_state=215, silent=False,
    )

    if 'TARGET' in df.columns:
        df = df.drop(['TARGET'], axis=1)

    factorize(df)

    train = pd.read_feather('./data/application_train.feather')
    train = train[['SK_ID_CURR', 'TARGET']]
    gc.collect()

    df = df.merge(train, on='SK_ID_CURR')
    y = df.pop('TARGET')

    embeded_lgb_selector = SelectFromModel(lgbc, threshold='{}*median'.format(thres))
    embeded_lgb_selector.fit(df, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = df.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features')
    if 'SK_ID_CURR' not in embeded_lgb_feature:
        embeded_lgb_feature.append('SK_ID_CURR')
    if 'TARGET' in orig.columns:
        embeded_lgb_feature.append('TARGET')
    print(embeded_lgb_feature, len(embeded_lgb_feature))
    return orig[embeded_lgb_feature]
