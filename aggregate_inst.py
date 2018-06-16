import pandas as pd
from utility import one_hot_encoder
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst():
    df = pd.read_feather('./data/installments_payments.preprocessed.feather')
    df, cat_columns = one_hot_encoder(df)

    key = ['SK_ID_CURR', 'SK_ID_PREV']
    # Features: Perform aggregations
    agg = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_columns:
        agg[cat] = ['mean']
    grp = df.groupby(key)
    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    ins_agg = g

    # Count installments accounts
    ins_agg['INS_COUNT'] = grp.size()
    ins_agg.columns = ["INS_" + c for c in ins_agg.columns]
    return ins_agg.reset_index()


def main():
    agg = aggregate_inst()
    agg.to_feather('./data/installments_payments.agg.feather')


if __name__ == '__main__':
    main()
