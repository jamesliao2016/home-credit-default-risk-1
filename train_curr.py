import os
import gc
import pickle
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
    df = pd.read_feather('./data/features.feather')
    print(df.shape)
    print('filter...')
    for fname in [
        './data/filter.0813-2134.pkl',
        './data/filter.0814-0834.pkl',
    ]:
        with open(fname, 'rb') as fp:
            filt = pickle.load(fp)
            df = df.drop(filt, axis=1)
    print(df.shape)
    print('split...')
    test = df[pd.isnull(df['TARGET'])].reset_index(drop=True)
    df = df[~pd.isnull(df['TARGET'])].reset_index(drop=True)
    train = df[~df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    valid = df[df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])].reset_index(drop=True)
    return train, valid, test


def train(idx):
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
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'max_bin': 300,
        'num_leaves': 30,
        'max_depth': -1,  # -1 means no limit
        'subsample': 1.0,
        'colsample_bytree': 0.05,
        # 'min_child_weight': 40,
        'min_child_samples': 70,
        'min_split_gain': 0.5,
        'reg_alpha': 0,
        'reg_lambda': 100,
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

    save_importance(bst, './data/importance.all.{}.csv'.format(idx))
    bst.save_model('./data/lgb.model.{}.{}.txt'.format(now, idx))

    test['PRED'] = bst.predict(test[features], bst.best_iteration)
    return test, score


def main():
    np.random.seed(215)

    debug = False
    print('debug: {}'.format(debug))
    gc.collect()

    auc_summary = []
    results = []
    for i in range(5):
        res, score = train(i)
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
    path = os.path.join('./output', '{}.csv.gz'.format(now))
    print('save {}'.format(path))
    res.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
