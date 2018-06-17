import pandas as pd
from aggregate_pos import aggregate_pos
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_pos_curr():
    df = pd.read_feather('./data/POS_CASH_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    del df['SK_ID_PREV']
    key = ['SK_ID_CURR']

    agg = aggregate_pos(df, key)

    # first/last
    grp = df.groupby(key)
    first_df = grp.first()
    first_df.columns = ['FIRST_{}'.format(c) for c in first_df.columns]
    agg = agg.join(first_df, on=key, how='left')
    last_df = grp.last()
    last_df.columns = ['LAST_{}'.format(c) for c in last_df.columns]
    agg = agg.join(last_df, on=key, how='left')
    agg.columns = ['POS_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    pos_agg = aggregate_pos_curr()
    pos_agg.to_feather('./data/POS_CASH_balance.agg.curr.feather')


if __name__ == '__main__':
    main()
