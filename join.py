import gc
import numpy as np
import pandas as pd
from utility import factorize, one_hot_encoder
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)


def add_bure_features(df):
    df['BURE_AMT_CREDIT_SUM_DEBT_SUM'].fillna(0, inplace=True)
    df['BURE_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    df['BURE_RATIO_CREDIT_DEBT'] = df['BURE_AMT_CREDIT_SUM_DEBT_SUM']
    df['BURE_RATIO_CREDIT_DEBT'] /= (1 + df['BURE_AMT_CREDIT_SUM_SUM'])
    df['BURE_RATIO_CREDIT_DEBT'] = df['BURE_RATIO_CREDIT_DEBT'].apply(np.tanh)

    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_AMT_CREDIT_SUM_OVERDUE_SUM']
    df['BURE_RATIO_CREDIT_OVERDUE'] /= (1 + df['BURE_AMT_CREDIT_SUM_DEBT_SUM'])
    df['BURE_RATIO_CREDIT_OVERDUE'] = df['BURE_RATIO_CREDIT_OVERDUE'].apply(np.tanh)

    df['BURE_ACT_DAYS_CREDIT_MAX'].fillna(-3000, inplace=True)

    df['BURE_ACT_AMT_ANNUITY_SUM'].fillna(0, inplace=True)
    df['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)

    return df


def add_inst_features(df):
    df['INS_AMT_PAYMENT_SUM'].fillna(0, inplace=True)
    df['INS_AMT_PAYMENT_SUM'] = df['INS_AMT_PAYMENT_SUM'].apply(np.tanh)

    df['INS_DPD_MEAN'].fillna(0, inplace=True)
    df['INS_DPD_MEAN'] = df['INS_DPD_MEAN'].apply(np.tanh)
    df['INS_DBD_MAX'].fillna(0, inplace=True)
    df['INS_DBD_MAX'] = df['INS_DBD_MAX'].apply(np.tanh)

    df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'].fillna(0, inplace=True)
    df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'] = df['INS_TSDIFF_DAYS_ENTRY_PAYMENT_STD'].apply(np.tanh)

    return df


def merge_bure(df):
    sum_bure = pd.read_feather('./data/bureau.agg.feather')
    df = df.merge(sum_bure, on='SK_ID_CURR', how='left')
    df = add_bure_features(df)

    return df


def merge_inst(df):
    sum_inst = pd.read_feather(
        './data/installments_payments.agg.curr.feather')
    df = df.merge(sum_inst, on='SK_ID_CURR', how='left')
    df = add_inst_features(df)

    return df


def merge_prev(df):
    print('merge prev...')
    prev = pd.read_feather('./data/previous_application.agg.feather')
    prev, _ = one_hot_encoder(prev)
    last = pd.read_feather('./data/previous_application.last.feather')
    last, _ = one_hot_encoder(last)
    df = df.merge(prev, on='SK_ID_CURR', how='left')
    df = df.merge(last, on='SK_ID_CURR', how='left')

    return df


def rename_columns(g, prefix):
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g.columns = ['{}_{}'.format(prefix, c) for c in g.columns]


def preprocess(debug):
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    if debug:
        train = train.sample(n=10000).reset_index(drop=True)
    test = pd.read_feather('./data/application_test.preprocessed.feather')

    df = pd.concat([train, test], sort=False)
    agg = ['mean', 'std', 'min', 'max', 'nunique']

    def summarize(df, prefix):
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

    print('summarize')
    sum_cred = get_cred()
    sum_pos = get_pos()
    last_inst = pd.read_feather(
        './data/installments_payments.agg.curr.last.feather')
    last_pos = pd.read_feather('./data/POS_CASH_balance.agg.curr.last.feather')
    last_cred = pd.read_feather(
        './data/credit_card_balance.agg.curr.last.feather')
    gc.collect()

    factorize(df)
    df = merge_bure(df)
    df = merge_prev(df)
    df = merge_inst(df)
    df = df.merge(sum_cred, on='SK_ID_CURR', how='left')
    df = df.merge(sum_pos, on='SK_ID_CURR', how='left')
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

    return df


def main():
    debug = False
    df = preprocess(debug)
    df.to_feather('./data/features.feather')


if __name__ == '__main__':
    main()
