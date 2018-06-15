import pandas as pd
from utility import one_hot_encoder
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_pos():
    df = pd.read_feather('./data/POS_CASH_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
    key = ['SK_ID_CURR', 'SK_ID_PREV']

    # count
    grp = df.groupby(key)
    g = grp[['MONTHS_BALANCE']].count()
    g.columns = ['COUNT']
    pos_agg = g

    # first/last
    first_df = grp.first()
    first_df.columns = ['FIRST_{}'.format(c) for c in first_df.columns]
    pos_agg = pos_agg.join(first_df, on=key, how='left')
    last_df = grp.last()
    last_df.columns = ['LAST_{}'.format(c) for c in last_df.columns]
    pos_agg = pos_agg.join(last_df, on=key, how='left')

    # aggregate
    df, cat_columns = one_hot_encoder(df)
    grp = df.groupby(key)
    fs = ['sum', 'median', 'mean', 'std', 'max', 'min']
    agg = {
        'CNT_INSTALMENT': fs,
        'CNT_INSTALMENT_FUTURE': fs,
        'SK_DPD': fs,
        'SK_DPD_DEF': fs,
        'DIFF_CNT_INSTALMENT': fs,
        'DIFF_CNT_INSTALMENT_FUTURE': fs,
        'DIFF_SK_DPD': fs,
        'DIFF_SK_DPD_DEF': fs,
    }
    for c in cat_columns:
        agg[c] = ['mean']
    g = grp.agg(agg)
    g.columns = ['{}_{}'.format(a.upper(), b) for a, b in g.columns]
    pos_agg = pos_agg.join(g, on=key, how='left')
    pos_agg.columns = ['POS_{}'.format(c) for c in pos_agg.columns]

    return pos_agg.reset_index()


def main():
    pos_agg = aggregate_pos()
    pos_agg.to_feather('./data/POS_CASH_balance.agg.feather')


if __name__ == '__main__':
    main()
