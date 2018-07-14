import gc
import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from utility import split_train, factorize
from train_curr import add_inst_features, add_bure_features
pd.set_option("display.max_columns", 500)


def train(idx, validate, importance_summay):
    src = './data/application_train.preprocessed.split.{}.feather'.format(idx)
    print('load {}'.format(src))
    train = pd.read_feather(src)
    if validate:
        train, test = split_train(train)

    else:
        test = pd.read_feather('./data/application_test.preprocessed.feather')

    train['IS_TEST'] = 0
    test['IS_TEST'] = 1
    df = pd.concat([train, test], sort=False)
    curr_id = df['SK_ID_CURR'].unique()

    def filter_by_curr(df):
        df = df[df['SK_ID_CURR'].isin(curr_id)].reset_index(drop=True)
        return df

    def join_bure(df):
        bure = pd.read_feather('./data/bureau.agg.feather')
        bure = filter_by_curr(bure)
        df = df.merge(bure, on='SK_ID_CURR', how='left')
        df = add_bure_features(df)
        return df

    def join_prev(df):
        prev = pd.read_feather('./data/previous_application.preprocessed.feather')
        prev = filter_by_curr(prev)
        prev = prev.set_index(['SK_ID_CURR', 'SK_ID_PREV'])
        prev.columns = ['PREV_{}'.format(c) for c in prev.columns]
        prev = prev.reset_index()
        df = df.merge(prev, on='SK_ID_CURR', how='left')
        df['PREV_AMT_ANNUITY'].fillna(0, inplace=True)
        df['PREV_AMT_CREDIT'].fillna(0, inplace=True)
        return df

    def join_cred(df):
        cred = pd.read_feather('./data/credit_card_balance.agg.prev.feather')
        cred = filter_by_curr(cred)
        df = df.merge(cred, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')

        last_cred = pd.read_feather('./data/credit_card_balance.agg.prev.last.feather')
        last_cred = filter_by_curr(last_cred)
        df = df.merge(last_cred, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')
        return df

    def join_inst(df):
        inst = pd.read_feather('./data/installments_payments.agg.prev.feather')
        inst = filter_by_curr(inst)
        df = df.merge(inst, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')
        df = add_inst_features(df)

        last_inst = pd.read_feather('./data/installments_payments.agg.prev.last.feather')
        last_inst = filter_by_curr(last_inst)
        df = df.merge(last_inst, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')
        return df

    def join_pos(df):
        pos = pd.read_feather('./data/POS_CASH_balance.agg.prev.feather')
        pos = filter_by_curr(pos)
        df = df.merge(pos, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')

        last_pos = pd.read_feather('./data/POS_CASH_balance.agg.prev.last.feather')
        last_pos = filter_by_curr(last_pos)
        df = df.merge(last_pos, on=['SK_ID_CURR', 'SK_ID_PREV'], how='outer')

        return df

    print('join bure')
    df = join_bure(df)
    print('join prev')
    df = join_prev(df)
    print('join credit')
    df = join_cred(df)
    print('join isnt')
    df = join_inst(df)
    print('join pos')
    df = join_pos(df)
    factorize(df)
    # agg = ['mean', 'std', 'min', 'max', 'nunique']

    # def summarize(df, prefix):
    #     df = df[df['SK_ID_CURR'].isin(curr_id)].reset_index(drop=True)
    #     factorize(df)
    #     if 'SK_ID_BUREAU' in df.columns:
    #         del df['SK_ID_BUREAU']
    #     if 'SK_ID_PREV' in df.columns:
    #         del df['SK_ID_PREV']
    #     res = df.groupby('SK_ID_CURR').agg(agg)
    #     rename_columns(res, prefix)
    #     return res

    gc.collect()

    # calculate length
    # df['ANNUITY_SUM_AP'] = df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY']
    # df['CREDIT_SUM_AP'] = df['AMT_CREDIT'] + df['PREV_AMT_CREDIT']
    # df['ANNUITY_SUM_LENGTH_AP'] = df['CREDIT_SUM_AP'] / df['ANNUITY_SUM_AP']
    # df['DIFF_ANNUITY_AND_INCOME_SUM_AP'] =\
    #     df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM_AP']

    # df['ANNUITY_SUM_AB'] = df['AMT_ANNUITY'] + df['BURE_ACT_AMT_ANNUITY_SUM']
    # df['CREDIT_SUM_AB'] = df['AMT_CREDIT'] + df['BURE_ACT_AMT_CREDIT_SUM_SUM']
    # df['ANNUITY_SUM_LENGTH_AB'] = df['CREDIT_SUM_AB'] / df['ANNUITY_SUM_AB']
    # df['DIFF_ANNUITY_AND_INCOME_SUM_AB'] =\
    #     df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM_AB']

    # df['ANNUITY_SUM'] = (
    #     df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY_SUM'] +
    #     df['BURE_ACT_AMT_ANNUITY_SUM'])
    # df['CREDIT_SUM'] = (
    #     df['AMT_CREDIT'] + df['PREV_AMT_CREDIT_SUM'] +
    #     df['BURE_ACT_AMT_CREDIT_SUM_SUM'])
    # df['ANNUITY_SUM_LENGTH'] = df['CREDIT_SUM'] / df['ANNUITY_SUM']
    # df['DIFF_ANNUITY_AND_INCOME_SUM'] =\
    #     df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM']

    # # TODO: mutate(na = apply(., 1, function(x) sum(is.na(x))),
    # # TODO: mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%
    # # TODO: mutate_all(funs(ifelse(is.infinite(.), NA, .))) %>%

    train = df[df['IS_TEST'] == 0].reset_index(drop=True)
    test = df[df['IS_TEST'] == 1].reset_index(drop=True)
    n_train = len(train)

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
    for c in [
        'SK_ID_CURR', 'SK_ID_PREV',
    ]:
        if c in features:
            features.remove(c)
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
        'num_leaves': 32,
        'max_depth': 8,  # -1 means no limit
        'subsample': 0.8715623,
        'colsample_bytree': 0.9497036,
        'min_child_weight': 40,
        'min_split_gain': 0.0222415,
        'reg_alpha': 0.04,
        'reg_lambda': 0.073,
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

    score = evals_result['valid']['auc'][bst.best_iteration-1]
    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", score)

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for key, value in zip(feature_name, importance):
        importance_summay[key] += value / sum(importance)

    if validate:
        test['TARGET'] = valid_y
    test['PRED'] = bst.predict(test[features], bst.best_iteration)
    test = test.groupby('SK_ID_CURR').mean().reset_index()
    if validate:
        score = roc_auc_score(test['TARGET'], test['PRED'])
        print('curr auc: {}'.format(score))
    return test, score


def main():
    np.random.seed(215)
    now = datetime.now().strftime('%m%d-%H%M')

    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    validate = args.validate
    print('validate: {}'.format(validate))
    if validate:
        n_bagging = 3
    else:
        n_bagging = 11
    importance_summay = defaultdict(lambda: 0)
    auc_summary = []
    results = []
    for i in range(n_bagging):
        res, score = train(i, validate, importance_summay)
        gc.collect()
        results.append(res)
        auc_summary.append(score)
        print('score: {}'.format(score))

    auc_summary = np.array(auc_summary)

    importances = list(sorted(importance_summay.items(), key=lambda x: -x[1]))
    for key, value in importances[:500]:
        print('{} {}'.format(key, value))

    print('validate auc: {} +- {}'.format(
        auc_summary.mean(), auc_summary.std()))

    if not validate:
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
