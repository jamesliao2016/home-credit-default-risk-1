import pandas as pd
from utility import reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_credit_last():
    df = pd.read_feather(
        './data/credit.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    last = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).last()
    last = last.reset_index()
    grp = last.groupby('SK_ID_CURR')

    g = grp.agg({
        'AMT_BALANCE': ['mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'],
        # added
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': ['sum'],
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': ['sum'],
    })

    g.columns = ['{}_{}'.format(a, b.upper()) for a, b in g.columns]
    agg = g

    agg.columns = ['CRED_PREV_LAST_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = aggregate_credit_last()
    reduce_memory(agg)
    agg.to_feather('./data/credit.prev.last.feather')


if __name__ == '__main__':
    main()
