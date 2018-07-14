import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst_prev_last():
    df = pd.read_feather('./data/installments_payments.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last.columns = [
        'INSTLAST_{}'.format(c) for c in last.columns]
    return last.reset_index()


def main():
    agg = aggregate_inst_prev_last()
    agg.to_feather(
        './data/installments_payments.agg.prev.last.feather')


if __name__ == '__main__':
    main()
