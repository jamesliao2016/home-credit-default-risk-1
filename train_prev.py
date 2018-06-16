import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
# from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from utility import one_hot_encoder, split_train
pd.set_option("display.max_columns", 500)


def split(df):
    pos_df = df[df['TARGET'] == 1].sample(frac=1)
    neg_df = df[df['TARGET'] == 0].sample(frac=1)
    n_pos = pos_df.shape[0]
    n_neg = neg_df.shape[0]
    n_pos_train = int(0.85*n_pos)
    n_neg_train = int(0.85*n_neg)
    train_df = pd.concat([pos_df[:n_pos_train], neg_df[:n_neg_train]])
    train_df = train_df.sample(frac=1).reset_index()
    test_df = pd.concat([pos_df[n_pos_train:], neg_df[n_neg_train:]])
    test_df = test_df.sample(frac=1).reset_index()
    return train_df, test_df


def preprocess_bureau(df):
    df['AMT_CREDIT_SUM'] = df['AMT_CREDIT'] + df['BURE_AMT_CREDIT_SUM_SUM']
    df['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    df['AMT_ANNUITY_SUM'] = df['AMT_ANNUITY'] + df['BURE_AMT_ANNUITY_SUM']
    df['AMT_ANNUITY_SUM'].fillna(0, inplace=True)
    df['RATIO_CREDIT_SUM_TO_ANNUITY_SUM'] =\
        df['AMT_CREDIT_SUM'] / df['AMT_ANNUITY_SUM']


def join_bureau(train_df, test_df):
    bur_df = pd.read_feather('./data/bureau.agg.feather')

    def f(df):
        key = ['SK_ID_CURR']
        df = df.merge(bur_df, on=key, how='left')
        preprocess_bureau(df)
        return df

    return f(train_df), f(test_df)


def join_prev(df, test_df):
    key = ['SK_ID_CURR']
    pre_df = pd.read_feather(
        './data/previous_application.joined.feather')
    pre_df = pre_df.set_index(['SK_ID_CURR', 'SK_ID_PREV'])
    pre_df.columns = ['PREV_{}'.format(c) for c in pre_df.columns]
    pre_df = pre_df.reset_index()
    df = df.merge(pre_df, on=key, how='left')
    test_df = test_df.merge(pre_df, on=key, how='left')

    return df, test_df


def join_inst(df, test_df):
    key = ['SK_ID_CURR', 'SK_ID_PREV']
    ins_df = pd.read_feather(
        './data/installments_payments.agg.feather')
    df = df.merge(ins_df, on=key, how='left')
    test_df = test_df.merge(ins_df, on=key, how='left')

    return df, test_df


def train(train_df, test_df, validate, importance_summay):
    print('join bureau...')
    train_df, test_df = join_bureau(train_df, test_df)
    gc.collect()
    print('join prev...')
    train_df, test_df = join_prev(train_df, test_df)
    gc.collect()
    print('join inst...')
    train_df, test_df = join_inst(train_df, test_df)
    gc.collect()

    n_train = len(train_df)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    del train_df
    del test_df
    gc.collect()
    df, _ = one_hot_encoder(df)
    test_df = df[n_train:].reset_index(drop=True)
    train_df = df[:n_train].reset_index(drop=True)

    # train
    if validate:
        n_train = len(train_df)
        train_df = train_df
        valid_df = test_df
    else:
        n_train = int(len(train_df) * 0.85)
        valid_df = train_df[n_train:]
        train_df = train_df[:n_train]

    features = train_df.columns.values.tolist()
    for t in ['SK_ID_CURR', 'SK_ID_PREV', 'TARGET', 'index']:
        while t in features:
            features.remove(t)

    xgtrain = lgb.Dataset(
        train_df[features].values, label=train_df['TARGET'].values,
        feature_name=features,
    )
    xgvalid = lgb.Dataset(
        valid_df[features].values, label=valid_df['TARGET'].values,
        feature_name=features,
    )
    evals_result = {}
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 7,
        'max_depth': -1,  # -1 means no limit
        'min_data_in_leaf': 40,
        'max_bin': 64,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'subsample_for_bin': 10000000,
        'min_split_gain': 0.01,
        'reg_alpha': 0.02,
        'reg_lambda': 0.02,
        'nthread': 12,
        'verbose': 0,
    }
    bst = lgb.train(
        lgb_params,
        xgtrain,
        valid_sets=[xgtrain, xgvalid],
        valid_names=['train', 'valid'],
        evals_result=evals_result,
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=50,
        categorical_feature=[],
        # feval=feval,
    )

    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", evals_result['valid']['auc'][bst.best_iteration-1])

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for key, value in zip(feature_name, importance):
        importance_summay[key] += value / sum(importance)

    test_df['PRED'] = bst.predict(test_df[features], bst.best_iteration)
    return test_df


def main():
    np.random.seed(215)
    # now = datetime.now().strftime('%m%d-%H%M')
    validate = True
    print('validate: {}'.format(validate))

    if validate:
        pass
    else:
        test_df = pd.read_feather(
            './data/application_test.preprocessed.feather')

    importance_summay = defaultdict(lambda: 0)
    auc_summary = []
    for i in range(1):
        src = './data/application_train.split.{}.feather'.format(i)
        df = pd.read_feather(src)
        if validate:
            train_df, test_df = split_train(df)
        else:
            train_df = df

        res_df = train(
            train_df, test_df, validate, importance_summay,
        )
        res_df = res_df.groupby('SK_ID_CURR')[['PRED']].mean().reset_index()
        test_df = test_df.merge(res_df, on='SK_ID_CURR', how='left')
        if validate:
            score = roc_auc_score(test_df['TARGET'], test_df['PRED'])
            auc_summary.append(score)
            print('score: {}'.format(score))

    auc_summary = np.array(auc_summary)
    importances = list(sorted(importance_summay.items(), key=lambda x: -x[1]))
    for key, value in importances[:500]:
        print('{} {}'.format(key, value))


if __name__ == '__main__':
    main()
