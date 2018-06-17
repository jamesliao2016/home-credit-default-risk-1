import pandas as pd
from aggregate_credit import aggregate_credit
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)


def aggregate_credit_by_prev_id():
    df = pd.read_feather(
        './data/credit_card_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    return aggregate_credit(df, ['SK_ID_CURR', 'SK_ID_PREV'])


def main():
    agg = aggregate_credit_by_prev_id()
    agg.to_feather('./data/credit_card_balance.agg.prev.feather')


if __name__ == '__main__':
    main()
