import pandas as pd
from aggregate_inst import aggregate_inst
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst_last():
    df = pd.read_feather(
        './data/installments_payments.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last = last.reset_index()
    del last['SK_ID_PREV']

    agg = aggregate_inst(last, ['SK_ID_CURR'])

    agg.columns = [
        'INSTLAST_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = aggregate_inst_last()
    agg.to_feather(
        './data/installments_payments.agg.curr.last.feather')


if __name__ == '__main__':
    main()
