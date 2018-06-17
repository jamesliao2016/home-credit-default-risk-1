import pandas as pd
from aggregate_credit import aggregate_credit
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)


def aggregate_credit_by_curr_id():
    df = pd.read_feather(
        './data/credit_card_balance.preprocessed.feather')
    del df['SK_ID_PREV']
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    agg = aggregate_credit(df, ['SK_ID_CURR'])

    # last
    g = df.groupby(['SK_ID_CURR']).last()
    g.columns = ['LAST_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    agg.columns = [
        'CRED_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = aggregate_credit_by_curr_id()
    agg.to_feather(
        './data/credit_card_balance.agg.curr.feather')


if __name__ == '__main__':
    main()
