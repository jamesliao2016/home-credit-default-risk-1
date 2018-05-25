import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
pd.set_option("display.max_columns", 100)


def join_pos_df(df, pos_df):
    grp = pos_df.groupby('SK_ID_CURR')
    columns = ['CNT_INSTALMENT_FUTURE']
    grp = grp[columns].mean()
    grp.columns = ['{}_mean'.format(c) for c in columns]
    grp = grp.reset_index()
    df = df.merge(grp, on='SK_ID_CURR', how='left')

    return df


def join_bure_df(df, bure_df):
    grp = bure_df.groupby('SK_ID_CURR')
    columns = ['DAYS_CREDIT']
    grp = grp[columns].mean()
    grp.columns = ['{}_mean'.format(c) for c in columns]
    grp = grp.reset_index()
    df = df.merge(grp, on='SK_ID_CURR', how='left')

    return df


def train(df, test_df, pos_df, bure_df):
    # filter by sample id
    sk_id_curr = pd.concat([df['SK_ID_CURR'], test_df['SK_ID_CURR']]).unique()
    pos_df = pos_df[pos_df['SK_ID_CURR'].isin(sk_id_curr)]
    bure_df = bure_df[bure_df['SK_ID_CURR'].isin(sk_id_curr)]

    features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH',   # Client's age in days at the time of application,time only relative to the application  # noqa
        'AMT_ANNUITY',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE',  # For consumer loans it is the price of the goods for which the loan is given  # noqa
    ]

    # POS
    df = join_pos_df(df, pos_df)
    features += [
        'CNT_INSTALMENT_FUTURE_mean',
    ]

    # credit bureau
    df = join_bure_df(df, bure_df)
    features += [
        'DAYS_CREDIT_mean',
    ]

    # train
    n_train = int(len(df) * 0.9)
    train_df = df[:n_train]
    valid_df = df[n_train:]

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
        'learning_rate': 0.1,
        'num_leaves': (2**3)-1,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,
        'max_bin': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'min_child_weight': 5,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        # 'scale_pos_weight': 200,
        'nthread': 12,
        'verbose': 0,
    }
    bst = lgb.train(
        lgb_params,
        xgtrain,
        valid_sets=[xgtrain, xgvalid],
        valid_names=['train', 'valid'],
        evals_result=evals_result,
        num_boost_round=1000,
        early_stopping_rounds=30,
        verbose_eval=10,
        # feval=feval,
    )

    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", evals_result['valid']['auc'][bst.best_iteration-1])

    test_df = join_pos_df(test_df, pos_df)
    test_df = join_bure_df(test_df, bure_df)
    return bst.predict(test_df[features], bst.best_iteration)


def split(df):
    pos_df = df[df['TARGET'] == 1]
    neg_df = df[df['TARGET'] == 0]
    n_pos = pos_df.shape[0]
    n_neg = neg_df.shape[0]
    n_pos_train = int(0.9*n_pos)
    n_neg_train = int(0.9*n_neg)
    train_df = pd.concat([pos_df[:n_pos_train], neg_df[:n_neg_train]])
    train_df = train_df.sample(frac=1).reset_index()
    test_df = pd.concat([pos_df[n_pos_train:], neg_df[n_neg_train:]])
    test_df = test_df.sample(frac=1).reset_index()
    return train_df, test_df


def main():
    validate = True
    train_df = pd.read_feather('./data/application_train.csv.feather')
    if validate:
        train_df, test_df = split(train_df)
    pos_train_df = train_df[train_df['TARGET'] == 1]
    neg_train_df = train_df[train_df['TARGET'] == 0]
    n_pos = pos_train_df.shape[0]
    n_bagging = 10
    pos_df = pd.read_feather('./data/POS_CASH_balance.csv.feather')
    bure_df = pd.read_feather('./data/bureau.csv.feather')
    for i in range(n_bagging):
        neg_part_train_df = neg_train_df.sample(n=n_pos)
        part_df = pd.concat([pos_train_df, neg_part_train_df])
        part_df = part_df.sample(frac=1)
        test_df['PRED_{}'.format(i)] = train(part_df, test_df, pos_df, bure_df)

    test_df['PRED'] = 0
    for i in range(n_bagging):
        test_df['PRED'] += test_df['PRED_{}'.format(i)]

    if validate:
        print('validate auc: {}'.format(
            roc_auc_score(test_df['TARGET'], test_df['PRED'])))


if __name__ == '__main__':
    main()
