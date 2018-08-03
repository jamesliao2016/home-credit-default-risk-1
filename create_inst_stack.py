import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from utility import factorize
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
'''
This script is based on https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-789
'''


def load(idx):
    print('merge...')
    inst = []
    for fname in [
        './data/inst.agg.feather',
        './data/inst.last.feather',
        './data/inst.tail.feather',
        './data/inst.diff.feather',
    ]:
        inst.append(pd.read_feather(fname).set_index('SK_ID_CURR'))
    inst = pd.concat(inst, axis=1)
    app = pd.read_feather('./data/app.agg.feather')
    test = pd.read_feather('./data/application_test.preprocessed.feather')
    test = test.merge(inst, on='SK_ID_CURR')
    test = test.merge(app, on='SK_ID_CURR')
    test['TARGET'] = np.nan
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    train = train.merge(inst, on='SK_ID_CURR')
    train = train.merge(app, on='SK_ID_CURR')
    del inst
    del app

    gc.collect()

    print('post process...')
    df = pd.concat([train, test])
    factorize(df)
    train = df[pd.notnull(df['TARGET'])].reset_index(drop=True)
    test = df[pd.isnull(df['TARGET'])].reset_index(drop=True)
    del df
    gc.collect()

    print('split...')
    fold = './data/fold.{}.feather'.format(idx)
    print('load {}'.format(fold))
    fold = pd.read_feather(fold)
    valid = train[train['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    train = train[~train['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    print('train: {}'.format(train.shape))
    print('valid: {}'.format(valid.shape))
    print('test: {}'.format(test.shape))
    return train, valid, test


def train(idx, importance_summay):
    train, valid, test = load(idx)
    gc.collect()

    print('train...')
    train_y = train.pop('TARGET')
    features = train.columns.values.tolist()
    features.remove('SK_ID_CURR')
    xgtrain = lgb.Dataset(
        train[features].values, label=train_y.values,
        feature_name=features,
        categorical_feature=[],
    )
    del train
    gc.collect()
    valid_y = valid.pop('TARGET')
    xgvalid = lgb.Dataset(
        valid[features].values, label=valid_y.values,
        feature_name=features,
        categorical_feature=[],
    )
    gc.collect()
    evals_result = {}
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 30,
        'max_depth': -1,  # -1 means no limit
        'subsample': 0.90,
        'colsample_bytree': 0.30,
        'min_child_weight': 40,
        # 'min_child_samples': 70,
        'min_split_gain': 0.5,
        'reg_alpha': 1,
        'reg_lambda': 10,
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

    valid['PRED'] = bst.predict(valid[features], bst.best_iteration)
    test['PRED'] = bst.predict(test[features], bst.best_iteration)
    return test, valid, score


def main():
    np.random.seed(215)

    importance_summay = defaultdict(lambda: 0)
    auc_summary = []
    results = []
    stack = []
    for i in range(5):
        res, valid, score = train(i, importance_summay)
        stack.append(valid[['SK_ID_CURR', 'PRED']])
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

    res = results[0][['SK_ID_CURR']].set_index('SK_ID_CURR').copy()
    res['PRED'] = 0
    for df in results:
        df = df.set_index('SK_ID_CURR')
        res['PRED'] += df['PRED']
    res['PRED'] /= res['PRED'].max()
    res = res.reset_index()
    stack.append(res)
    stack = pd.concat(stack).reset_index(drop=True)
    stack = stack.rename(columns={'PRED': 'INST_STACKED'})

    print(stack.shape)
    print(stack['SK_ID_CURR'].unique().shape)

    stack.to_feather('./data/inst.stack.feather')


if __name__ == '__main__':
    main()
