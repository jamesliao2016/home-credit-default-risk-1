import pandas as pd
from utility import reduce_memory


def _aggregate():
    df = pd.read_feather('./data/inst.preprocessed.feather')
    agg = {
        # original
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['sum', 'max', 'min'],
        'DAYS_INSTALMENT': ['sum', 'max', 'min'],
        # preprocessed
        'RATIO_PAYMENT': ['max', 'min', 'mean', 'std'],
        'DIFF_PAYMENT': ['max', 'min', 'mean', 'sum', 'std'],
        'FLAG_DIFF_PAYMENT': ['mean'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'IS_CREDIT': ['mean'],
        'FLAG_DPD': ['mean'],
        'FLAG_DBD': ['mean'],
    }

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
    reduce_memory(agg)
    agg.to_feather('./data/inst.agg.feather')


if __name__ == '__main__':
    main()
