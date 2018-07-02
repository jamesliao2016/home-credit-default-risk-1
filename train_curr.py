import os
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from utility import split_train
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
'''
This script is based on https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-789
'''


def merge_bure(df):
    sum_bure = pd.read_feather('./data/bureau.agg.feather')
    df = df.merge(sum_bure, on='SK_ID_CURR', how='left')
    df['BURE_AMT_CREDIT_SUM_DEBT_SUM'].fillna(0, inplace=True)
    df['BURE_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    df['BURE_RATIO_CREDIT_DEBT'] = df['BURE_AMT_CREDIT_SUM_DEBT_SUM']
    df['BURE_RATIO_CREDIT_DEBT'] /= (1 + df['BURE_AMT_CREDIT_SUM_SUM'])
    df['BURE_RATIO_CREDIT_DEBT'] = df['BURE_RATIO_CREDIT_DEBT'].apply(np.tanh)

    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_AMT_CREDIT_SUM_OVERDUE_SUM']
    df['BURE_RATIO_CREDIT_OVERDUE'] /= (1 + df['BURE_AMT_CREDIT_SUM_DEBT_SUM'])
    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_RATIO_CREDIT_OVERDUE'].apply(np.tanh)

    df['BURE_ACT_DAYS_CREDIT_MAX'].fillna(-3000, inplace=True)

    return df


def merge_inst(df):
    sum_inst = pd.read_feather(
        './data/installments_payments.agg.curr.feather')
    df = df.merge(sum_inst, on='SK_ID_CURR', how='left')

    df['INS_AMT_PAYMENT_SUM'].fillna(0, inplace=True)
    df['INS_AMT_PAYMENT_SUM'] = df['INS_AMT_PAYMENT_SUM'].apply(np.tanh)

    df['INS_DPD_MEAN'].fillna(0, inplace=True)
    df['INS_DPD_MEAN'] = df['INS_DPD_MEAN'].apply(np.tanh)
    df['INS_DBD_MAX'].fillna(0, inplace=True)
    df['INS_DBD_MAX'] = df['INS_DBD_MAX'].apply(np.tanh)

    df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'].fillna(0, inplace=True)
    df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'] = df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'].apply(np.tanh)

    return df


def merge_agg_app(df):
    columns = ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'CODE_GENDER']
    agg = pd.read_feather('./data/application.agg.feather')
    df = df.merge(agg, on=columns, how='left')
    return df


def factorize(df):
    columns = df.select_dtypes([np.object]).columns.tolist()
    for c in columns:
        df[c], _ = pd.factorize(df[c])


def rename_columns(g, prefix):
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g.columns = ['{}_{}'.format(prefix, c) for c in g.columns]


def train(idx, validate, importance_summay):
    src = './data/application_train.preprocessed.split.{}.feather'.format(idx)
    print('load {}'.format(src))
    train = pd.read_feather(src)
    if validate:
        train, test = split_train(train)
    else:
        test = pd.read_feather('./data/application_test.preprocessed.feather')

    n_train = len(train)
    df = pd.concat([train, test], sort=False)
    agg = ['mean', 'std', 'min', 'max', 'nunique']

    curr_id = df['SK_ID_CURR'].unique()

    def summarize(df, prefix):
        df = df[df['SK_ID_CURR'].isin(curr_id)].reset_index(drop=True)
        factorize(df)
        if 'SK_ID_BUREAU' in df.columns:
            del df['SK_ID_BUREAU']
        if 'SK_ID_PREV' in df.columns:
            del df['SK_ID_PREV']
        res = df.groupby('SK_ID_CURR').agg(agg)
        rename_columns(res, prefix)
        return res

    def get_cred():
        cred = pd.read_feather(
            './data/credit_card_balance.agg.curr.feather')
        factorize(cred)
        return cred

    def get_pos():
        pos = pd.read_feather('./data/POS_CASH_balance.agg.curr.feather')
        factorize(pos)
        return pos

    def get_prev():
        prev = pd.read_feather('./data/previous_application.agg.feather')
        return prev

    print('summarize')
    sum_cred = get_cred()
    sum_pos = get_pos()
    sum_prev = get_prev()
    last_inst = pd.read_feather(
        './data/installments_payments.agg.curr.last.feather')
    last_pos = pd.read_feather('./data/POS_CASH_balance.agg.curr.last.feather')
    last_cred = pd.read_feather(
        './data/credit_card_balance.agg.curr.last.feather')
    gc.collect()

    df = merge_agg_app(df)
    factorize(df)
    df = merge_bure(df)
    df = merge_inst(df)
    df = df.merge(sum_cred, on='SK_ID_CURR', how='left')
    df = df.merge(sum_pos, on='SK_ID_CURR', how='left')
    df = df.merge(sum_prev, on='SK_ID_CURR', how='left')
    df = df.merge(last_inst, on='SK_ID_CURR', how='left')
    df = df.merge(last_pos, on='SK_ID_CURR', how='left')
    df = df.merge(last_cred, on='SK_ID_CURR', how='left')

    # fillna
    df['PREV_AMT_ANNUITY_SUM'].fillna(0, inplace=True)
    df['PREV_AMT_CREDIT_SUM'].fillna(0, inplace=True)
    df['BURE_ACT_AMT_ANNUITY_SUM'].fillna(0, inplace=True)
    df['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    # calculate length
    df['ANNUITY_SUM_AP'] = df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY_SUM']
    df['CREDIT_SUM_AP'] = df['AMT_CREDIT'] + df['PREV_AMT_CREDIT_SUM']
    df['ANNUITY_SUM_LENGTH_AP'] = df['CREDIT_SUM_AP'] / df['ANNUITY_SUM_AP']
    df['DIFF_ANNUITY_AND_INCOME_SUM_AP'] =\
        df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM_AP']

    df['ANNUITY_SUM_AB'] = df['AMT_ANNUITY'] + df['BURE_ACT_AMT_ANNUITY_SUM']
    df['CREDIT_SUM_AB'] = df['AMT_CREDIT'] + df['BURE_ACT_AMT_CREDIT_SUM_SUM']
    df['ANNUITY_SUM_LENGTH_AB'] = df['CREDIT_SUM_AB'] / df['ANNUITY_SUM_AB']
    df['DIFF_ANNUITY_AND_INCOME_SUM_AB'] =\
        df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM_AB']

    df['ANNUITY_SUM'] = (
        df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY_SUM'] +
        df['BURE_ACT_AMT_ANNUITY_SUM'])
    df['CREDIT_SUM'] = (
        df['AMT_CREDIT'] + df['PREV_AMT_CREDIT_SUM'] +
        df['BURE_ACT_AMT_CREDIT_SUM_SUM'])
    df['ANNUITY_SUM_LENGTH'] = df['CREDIT_SUM'] / df['ANNUITY_SUM']
    df['DIFF_ANNUITY_AND_INCOME_SUM'] =\
        df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM']

    # TODO: mutate(na = apply(., 1, function(x) sum(is.na(x))),
    # TODO: mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%
    # TODO: mutate_all(funs(ifelse(is.infinite(.), NA, .))) %>%

    train = df[:n_train].reset_index(drop=True)
    test = df[n_train:].reset_index(drop=True)

    if validate:
        n_train = len(train)
        train = train
        valid = test
    else:
        n_train = int(len(train) * 0.85)
        valid = train[n_train:].reset_index(drop=True)
        train = train[:n_train].reset_index(drop=True)

    train_y = train.pop('TARGET')
    valid_y = valid.pop('TARGET')
    features = train.columns.values.tolist()
    features.remove('SK_ID_CURR')
    xgtrain = lgb.Dataset(
        train[features].values, label=train_y.values,
        feature_name=features,
        categorical_feature=[],
    )
    xgvalid = lgb.Dataset(
        valid[features].values, label=valid_y.values,
        feature_name=features,
        categorical_feature=[],
    )
    evals_result = {}
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': -1,  # -1 means no limit
        # 'min_data_in_leaf': 40,
        # 'max_bin': 64,
        'subsample': 0.7,
        # 'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'min_child_weight': 120,
        # 'subsample_for_bin': 10000000,
        'min_split_gain': 0.00,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
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
        early_stopping_rounds=200,
        verbose_eval=50,
        categorical_feature=[],
        # feval=feval,
    )

    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", evals_result['valid']['auc'][bst.best_iteration-1])

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for key, value in zip(feature_name, importance):
        importance_summay[key] += value / sum(importance)

    if validate:
        test['TARGET'] = valid_y
    test['PRED'] = bst.predict(test[features], bst.best_iteration)
    return test


def main():
    np.random.seed(215)
    now = datetime.now().strftime('%m%d-%H%M')

    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    validate = args.validate
    print('validate: {}'.format(validate))
    if validate:
        n_bagging = 5
    else:
        n_bagging = 11
    importance_summay = defaultdict(lambda: 0)
    auc_summary = []
    results = []
    for i in range(n_bagging):
        res = train(i, validate, importance_summay)
        results.append(res)
        if validate:
            score = roc_auc_score(res['TARGET'], res['PRED'])
            auc_summary.append(score)
            print('score: {}'.format(score))

    auc_summary = np.array(auc_summary)

    importances = list(sorted(importance_summay.items(), key=lambda x: -x[1]))
    for key, value in importances[:500]:
        print('{} {}'.format(key, value))

    if validate:
        print('validate auc: {} +- {}'.format(
            auc_summary.mean(), auc_summary.std()))
    else:
        res = results[0][['SK_ID_CURR']].set_index('SK_ID_CURR')
        res['TARGET'] = 0
        for df in results:
            df = df.set_index('SK_ID_CURR')
            res['TARGET'] += df['PRED']
        res['TARGET'] /= res['TARGET'].max()
        res = res.reset_index()
        os.makedirs('./output', exist_ok=True)
        path = os.path.join('./output', '{}.csv.gz'.format(now))
        print('save {}'.format(path))
        res.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
