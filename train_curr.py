import os
import gc
import pickle
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from utility import save_importance
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
now = datetime.now().strftime('%m%d-%H%M')


def load(idx):
    fold = './data/fold.{}.feather'.format(idx)
    print('load {}'.format(fold))
    fold = pd.read_feather(fold)
    print('load features...')
    df = pd.read_feather('./data/features.feather').set_index('SK_ID_CURR')
    nn = pd.read_feather('./data/nn.feather').set_index('SK_ID_CURR')
    df['nn'] = nn['nn_0']
    del nn
    df = df.reset_index()
    print(df.shape)
    print('filter...')
    for fname in [
    ]:
        with open(fname, 'rb') as fp:
            filt = pickle.load(fp)
            df = df.drop(filt, axis=1)
    for c in df.columns:
        if str(df[c].dtype) == 'category':
            df[c] = df[c].astype('int16')
    print(df.shape)
    print('split...')
    test = df[pd.isnull(df['TARGET'])].reset_index(drop=True)
    df = df[~pd.isnull(df['TARGET'])].reset_index(drop=True)
    train = df[~df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    valid = df[df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    print('valid: {}'.format(valid.shape))
    return train, valid, test


def train(idx, seed, param_idx):
    print('seed {}, param_idx {}'.format(seed, param_idx))
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
    del valid
    gc.collect()
    evals_result = {}
    params = [
        {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.02,
            'max_bin': 300,
            'num_leaves': 30,
            'max_depth': -1,  # -1 means no limit
            'subsample': 1.0,
            'colsample_bytree': 0.70,
            'min_child_weight': 200,
            'min_child_samples': 70,
            'min_split_gain': 0.5,
            'reg_alpha': 0,
            'reg_lambda': 100,
            'nthread': 12,
            'verbose': 0,
            'seed': seed,
        },
        {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 12,
            'learning_rate': 0.02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,
            'verbose': -1,
            'metric': 'auc',
            'seed': seed,
        }
    ]
    bst = lgb.train(
        params[param_idx],
        xgtrain,
        valid_sets=[xgtrain, xgvalid],
        valid_names=['train', 'valid'],
        evals_result=evals_result,
        num_boost_round=3000,
        early_stopping_rounds=200,
        verbose_eval=50,
        categorical_feature=[],
        # feval=feval,
    )

    score = evals_result['valid']['auc'][bst.best_iteration-1]
    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", score)

    save_importance(bst, './data/importance.all.{}.csv'.format(idx))
    bst.save_model('./data/lgb.model.{}.{}.txt'.format(now, idx))

    test['PRED'] = bst.predict(test[features], bst.best_iteration)
    return test, score


def main():
    np.random.seed(215)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--param-idx', required=True, type=int)
    args = parser.parse_args()

    debug = False
    print('debug: {}'.format(debug))
    gc.collect()

    auc_summary = []
    results = []
    for i in range(5):
        res, score = train(i, args.seed, args.param_idx)
        gc.collect()
        results.append(res)
        auc_summary.append(score)
        print('score: {}'.format(score))

    auc_summary = np.array(auc_summary)

    print('validate auc: {} +- {}'.format(
        auc_summary.mean(), auc_summary.std()))

    res = results[0][['SK_ID_CURR']].set_index('SK_ID_CURR')
    res['TARGET'] = 0
    for df in results:
        df = df.set_index('SK_ID_CURR')
        res['TARGET'] += df['PRED']
    res['TARGET'] /= res['TARGET'].max()
    res = res.reset_index()
    os.makedirs('./output', exist_ok=True)
    path = os.path.join(
        './output', '{}.seed.{}.param.{}.csv.gz'.format(now, args.seed, args.param_idx))
    print('save {}'.format(path))
    res.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
