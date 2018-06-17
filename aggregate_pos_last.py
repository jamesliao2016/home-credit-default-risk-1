import pandas as pd
from aggregate_pos import aggregate_pos


def aggregate_pos_last():
    df = pd.read_feather('./data/POS_CASH_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last = last.reset_index()
    del last['SK_ID_PREV']

    agg = aggregate_pos(last, ['SK_ID_CURR'])

    agg.columns = ['POSLAST_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = aggregate_pos_last()
    agg.to_feather('./data/POS_CASH_balance.agg.curr.last.feather')


if __name__ == '__main__':
    main()
