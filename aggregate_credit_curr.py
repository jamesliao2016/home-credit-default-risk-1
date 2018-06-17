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

    return aggregate_credit(df, ['SK_ID_CURR'])


def main():
    agg = aggregate_credit_by_curr_id()
    agg.to_feather(
        './data/credit_card_balance.agg.curr.feather')


if __name__ == '__main__':
    main()
