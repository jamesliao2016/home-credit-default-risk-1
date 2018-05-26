import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from sklearn.metrics import roc_auc_score
pd.set_option("display.max_columns", 100)


def join_pos_df(df, test_df, pos_df, features):
    # TODO: recent balance
    grp = pos_df.groupby('SK_ID_CURR')
    for agg, columns in [
        [
            'count', []
        ],
        [
            'mean', [
                'CNT_INSTALMENT_FUTURE',
            ],
        ],
    ]:
        if agg == 'count':
            g = grp[['SK_ID_PREV']].count()
        elif agg == 'mean':
            g = grp[columns].mean()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))

        if agg == 'count':
            columns = ['pos_COUNT']
        else:
            columns = ['pos_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_bure_df(df, test_df, bure_df, features):
    grp = bure_df.groupby('SK_ID_CURR')
    for agg, columns in {
        'mean': [
            'DAYS_CREDIT',  # How many days before current application did client apply for Credit Bureau credit,time only relative to the application  # noqa
            'AMT_CREDIT_SUM',  # Current credit amount for the Credit Bureau credit  # noqa
            'AMT_CREDIT_SUM_DEBT',  # Current debt on Credit Bureau credit
        ],
        'sum': [
            'AMT_CREDIT_SUM_DEBT',  # Current debt on Credit Bureau credit
        ],
        'max': [
            'DAYS_CREDIT',
            'AMT_CREDIT_SUM',  # Current credit amount for the Credit Bureau credit  # noqa
        ],
    }.items():
        if agg == 'mean':
            g = grp[columns].mean()
        elif agg == 'max':
            g = grp[columns].max()
        elif agg == 'sum':
            g = grp[columns].sum()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['bureau_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_credit_df(df, test_df, credit_df, features):
    # TODO: recent credit
    grp = credit_df.groupby('SK_ID_CURR')
    for agg, columns in [
        ['mean', [
            'SK_DPD',
            'SK_DPD_DEF',
        ]],
    ]:
        if agg == 'mean':
            g = grp[columns].mean()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['credit_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def join_prev_df(df, test_df, prev_df, features):
    grp = prev_df.groupby('SK_ID_CURR')
    for agg, columns in [
        [
            'mean', [
                'CNT_PAYMENT',  # Term of previous credit at application of the previous application  # noqa
            ],
        ],
    ]:
        if agg == 'mean':
            g = grp[columns].mean()
        else:
            raise RuntimeError('agg is invalid {}'.format(agg))
        columns = ['prev_{}_{}'.format(c, agg) for c in columns]
        g.columns = columns
        features += columns
        g = g.reset_index()
        df = df.merge(g, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(g, on='SK_ID_CURR', how='left')

    return df, test_df, features


def train(df, test_df, pos_df, bure_df, credit_df, prev_df, importance_summay):
    # filter by sample id
    sk_id_curr = pd.concat([df['SK_ID_CURR'], test_df['SK_ID_CURR']]).unique()
    pos_df = pos_df[pos_df['SK_ID_CURR'].isin(sk_id_curr)]
    bure_df = bure_df[bure_df['SK_ID_CURR'].isin(sk_id_curr)]
    prev_df = prev_df[prev_df['SK_ID_CURR'].isin(sk_id_curr)]

    features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH',   # Client's age in days at the time of application,time only relative to the application  # noqa
        'AMT_ANNUITY',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE',  # For consumer loans it is the price of the goods for which the loan is given  # noqa
        'DAYS_EMPLOYED',  # How many days before the application the person started current employment,time only relative to the application  # noqa
    ]

    # POS
    df, test_df, features = join_pos_df(df, test_df, pos_df, features)

    # credit bureau
    df, test_df, features = join_bure_df(df, test_df, bure_df, features)

    # credit card
    df, test_df, features = join_credit_df(df, test_df, credit_df, features)

    # prev_df
    df, test_df, features = join_prev_df(df, test_df, prev_df, features)

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
        'num_leaves': (2**5)-1,  # 2^max_depth - 1
        'max_depth': 5,  # -1 means no limit
        'min_child_samples': 100,
        'max_bin': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'min_child_weight': 2,
        'subsample_for_bin': 200000,
        'min_split_gain': 0.01,
        'reg_alpha': 0,
        'reg_lambda': 0,
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
        verbose_eval=20,
        # feval=feval,
    )

    print("\nModel Report")
    print("bst1.best_iteration: ", bst.best_iteration)
    print("auc:", evals_result['valid']['auc'][bst.best_iteration-1])

    importance = bst.feature_importance(iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for key, value in zip(feature_name, importance):
        importance_summay[key] += value / sum(importance)

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
        n_bagging = 5
        train_df, test_df = split(train_df)
    pos_train_df = train_df[train_df['TARGET'] == 1]
    neg_train_df = train_df[train_df['TARGET'] == 0]
    n_pos = pos_train_df.shape[0]
    importance_summay = defaultdict(lambda: 0)
    pos_df = pd.read_feather('./data/POS_CASH_balance.csv.feather')
    bure_df = pd.read_feather('./data/bureau.csv.feather')
    credit_df = pd.read_feather('./data/credit_card_balance.csv.feather')
    prev_df = pd.read_feather('./data/previous_application.csv.feather')
    for i in range(n_bagging):
        neg_part_train_df = neg_train_df.sample(n=n_pos)
        part_df = pd.concat([pos_train_df, neg_part_train_df])
        part_df = part_df.sample(frac=1)
        test_df['PRED_{}'.format(i)] = train(
            part_df, test_df, pos_df, bure_df, credit_df, prev_df,
            importance_summay)

    for key, value in sorted(importance_summay.items(), key=lambda x: -x[1]):
        print('{} {}'.format(key, value))

    test_df['PRED'] = 0
    for i in range(n_bagging):
        test_df['PRED'] += test_df['PRED_{}'.format(i)]

    if validate:
        print('validate auc: {}'.format(
            roc_auc_score(test_df['TARGET'], test_df['PRED'])))


if __name__ == '__main__':
    main()
