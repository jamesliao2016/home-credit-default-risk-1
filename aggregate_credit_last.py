import pandas as pd
from aggregate_credit import aggregate_credit
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_credit_last():
    df = pd.read_feather(
        './data/credit_card_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last = last.reset_index()
    del last['SK_ID_PREV']

    agg = aggregate_credit(last, ['SK_ID_CURR'])
    agg.columns = [
        'CREDLAST_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = aggregate_credit_last()
    agg.to_feather(
        './data/credit_card_balance.agg.curr.last.feather')


if __name__ == '__main__':
    main()
