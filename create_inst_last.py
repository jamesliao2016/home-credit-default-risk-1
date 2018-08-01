import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst_last():
    df = pd.read_feather('./data/inst.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])

    last = df.groupby(['SK_ID_CURR']).last()
    last.columns = ['INST_LAST_{}'.format(c) for c in last.columns]

    return last.reset_index()


def main():
    agg = aggregate_inst_last()
    agg.to_feather('./data/inst.last.feather')


if __name__ == '__main__':
    main()
