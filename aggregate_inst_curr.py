import pandas as pd
from aggregate_inst import aggregate_inst
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst_curr():
    df = pd.read_feather('./data/installments_payments.preprocessed.feather')
    del df['SK_ID_PREV']
    df = df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])

    key = ['SK_ID_CURR']

    agg = aggregate_inst(df, key)

    # last
    g = df.groupby(key).last()
    g.columns = ['LAST_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    agg.columns = ["INS_" + c for c in agg.columns]
    return agg.reset_index()


def main():
    agg = aggregate_inst_curr()
    agg.to_feather('./data/installments_payments.agg.curr.feather')


if __name__ == '__main__':
    main()
