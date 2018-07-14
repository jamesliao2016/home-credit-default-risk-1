import pandas as pd


def aggregate_pos_last():
    df = pd.read_feather('./data/POS_CASH_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last.columns = ['POSLAST_{}'.format(c) for c in last.columns]
    return last.reset_index()


def main():
    agg = aggregate_pos_last()
    agg.to_feather('./data/POS_CASH_balance.agg.prev.last.feather')


if __name__ == '__main__':
    main()
