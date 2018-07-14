import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_credit_prev_last():
    df = pd.read_feather(
        './data/credit_card_balance.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last.columns = [
        'CREDLAST_{}'.format(c) for c in last.columns]
    return last.reset_index()


def main():
    agg = aggregate_credit_prev_last()
    agg.to_feather(
        './data/credit_card_balance.agg.prev.last.feather')


if __name__ == '__main__':
    main()
