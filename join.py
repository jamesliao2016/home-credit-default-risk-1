import gc
import numpy as np
import pandas as pd
from utility import factorize
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)


def add_bure_features(df):
    df['BURE_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_AMT_CREDIT_SUM_OVERDUE_SUM']
    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_RATIO_CREDIT_OVERDUE'].apply(np.tanh)

    df['BURE_ACT_DAYS_CREDIT_MAX'].fillna(-3000, inplace=True)

    df['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    return df


def add_inst_features(df):
    df['INS_AMT_PAYMENT_SUM'].fillna(0, inplace=True)
    df['INS_AMT_PAYMENT_SUM'] = df['INS_AMT_PAYMENT_SUM'].apply(np.tanh)

    df['INS_DPD_MEAN'].fillna(0, inplace=True)
    df['INS_DPD_MEAN'] = df['INS_DPD_MEAN'].apply(np.tanh)
    df['INS_DBD_MAX'].fillna(0, inplace=True)
    df['INS_DBD_MAX'] = df['INS_DBD_MAX'].apply(np.tanh)

    return df


def merge_app(df):
    print('merge app...')
    app = pd.read_feather('./data/app.agg.feather')
    df = df.merge(app, on='SK_ID_CURR', how='left')

    return df


def merge_bure(df):
    for fname in [
        './data/bureau.agg.feather',
    ]:
        print('merge {}...'.format(fname))
        bure = pd.read_feather(fname)
        df = df.merge(bure, on='SK_ID_CURR', how='left')
    df = add_bure_features(df)

    return df


def merge_inst(df):
    for fname in [
        './data/inst.agg.feather',
        './data/inst.last.feather',
        './data/inst.tail.feather',
        './data/inst.diff.feather',
        './data/inst.trend.feather',
    ]:
        inst = pd.read_feather(fname)
        df = df.merge(inst, on='SK_ID_CURR', how='left')
        print('merge {}...'.format(fname))

    df = add_inst_features(df)

    return df


def merge_prev(df):
    for fname in [
        './data/prev.agg.feather',
        './data/prev.refused.feather',
        './data/prev.approved.feather',
        './data/prev.last.feather',
        './data/prev.grp.feather',
    ]:
        print('merge {}...'.format(fname))
        prev = pd.read_feather(fname)
        df = df.merge(prev, on='SK_ID_CURR', how='left')

    return df


def merge_cred(df):
    for fname in [
        './data/credit.agg.feather',
        './data/credit.last.feather',
        './data/credit.prev.last.feather',
        './data/credit.diff.feather',
        './data/credit.tail.feather',
    ]:
        print('merge {}...'.format(fname))
        cred = pd.read_feather(fname)
        df = df.merge(cred, on='SK_ID_CURR', how='left')
    return df


def merge_pos(df):
    for fname in [
        './data/pos.agg.feather',
        './data/pos.diff.feather',
        './data/pos.tail.feather',
        './data/pos.trend.feather',
        './data/pos.last.feather',
    ]:
        gc.collect()
        print('merge {}...'.format(fname))
        pos = pd.read_feather(fname)
        df = df.merge(pos, on='SK_ID_CURR', how='left')
    return df


def rename_columns(g, prefix):
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g.columns = ['{}_{}'.format(prefix, c) for c in g.columns]


def post_process(df):
    factorize(df)

    # fillna
    df['PREV_AMT_ANNUITY_SUM'].fillna(0, inplace=True)
    df['PREV_AMT_CREDIT_SUM'].fillna(0, inplace=True)
    df['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    # calculate length
    df['ANNUITY_SUM_AP'] = df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY_SUM']
    df['CREDIT_SUM_AP'] = df['AMT_CREDIT'] + df['PREV_AMT_CREDIT_SUM']
    df['ANNUITY_SUM_LENGTH_AP'] = df['CREDIT_SUM_AP'] / df['ANNUITY_SUM_AP']
    df['DIFF_ANNUITY_AND_INCOME_SUM_AP'] =\
        df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM_AP']

    df['CREDIT_SUM_AB'] = df['AMT_CREDIT'] + df['BURE_ACT_AMT_CREDIT_SUM_SUM']

    df['ANNUITY_SUM'] = df['AMT_ANNUITY'] + df['PREV_AMT_ANNUITY_SUM']
    df['CREDIT_SUM'] = (
        df['AMT_CREDIT'] + df['PREV_AMT_CREDIT_SUM'] +
        df['BURE_ACT_AMT_CREDIT_SUM_SUM'])
    df['ANNUITY_SUM_LENGTH'] = df['CREDIT_SUM'] / df['ANNUITY_SUM']
    df['DIFF_ANNUITY_AND_INCOME_SUM'] =\
        df['AMT_INCOME_TOTAL'] - df['ANNUITY_SUM']

    df['COUNT_NAN'] = df.isnull().sum(axis=1)
    with pd.option_context('mode.use_inf_as_na', True):
        df['COUNT_INF'] = df.isnull().sum(axis=1) - df['COUNT_NAN']
    # TODO: mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%

    return df


def preprocess(debug):
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    if debug:
        train = train.sample(n=10000).reset_index(drop=True)
    test = pd.read_feather('./data/application_test.preprocessed.feather')

    df = pd.concat([train, test], sort=False)

    df = merge_app(df)
    gc.collect()
    df = merge_bure(df)
    gc.collect()
    df = merge_prev(df)
    gc.collect()
    df = merge_inst(df)
    gc.collect()
    df = merge_cred(df)
    gc.collect()
    df = merge_pos(df)
    gc.collect()

    df = post_process(df)

    return df


def main():
    debug = False
    df = preprocess(debug)
    df.to_feather('./data/features.feather')


if __name__ == '__main__':
    main()
