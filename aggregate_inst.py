import pandas as pd
from utility import reduce_memory, filter_by_lgb


def _aggregate():
    df = pd.read_feather('./data/inst.preprocessed.feather')
    # Features: Perform aggregations
    agg = {}
    for c in df.columns:
        if df[c].dtype == 'object':
            agg[c] = ['nunique', 'mode']
        else:
            agg[c] = ['min', 'max', 'mean', 'sum', 'std']

    grp = df.groupby('SK_ID_CURR')

    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g['COUNT'] = grp.size()
    agg = g

    agg['AMT_DIFF_PAYMENT_INSTALMENT'] = agg['AMT_INSTALMENT_SUM'] - agg['AMT_PAYMENT_SUM']
    agg['DAYS_DIFF_PAYMENT_INSTALMENT'] = agg['DAYS_INSTALMENT_SUM'] - agg['DAYS_ENTRY_PAYMENT_SUM']
    agg.columns = ["INS_" + c for c in agg.columns]

    return agg.reset_index()


def main():
    agg = _aggregate()
    agg = filter_by_lgb(agg)
    reduce_memory(agg)
    agg.to_feather('./data/inst.agg.feather')


if __name__ == '__main__':
    main()
