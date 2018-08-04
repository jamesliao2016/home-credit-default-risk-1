import pandas as pd
from utility import reduce_memory
from concurrent.futures import ProcessPoolExecutor


def _tail(k):
    print('create credit tail {}...'.format(k))
    df = pd.read_feather('./data/credit.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df = df.groupby('SK_ID_CURR').tail(k).reset_index(drop=True)
    grp = df.groupby('SK_ID_CURR')

    agg = {
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['sum', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['sum', 'mean'],
        'AMT_INST_MIN_REGULARITY': ['sum', 'mean'],
        'SK_DPD': ['mean'],
        # added
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': ['sum', 'mean'],
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': ['sum', 'mean'],
    }
    g = grp.agg(agg)
    g.columns = ['{}_{}'.format(a, b.upper()) for a, b in g.columns]
    g['COUNT'] = grp.size()
    g.columns = ['CRED_TAIL_{}_{}'.format(k, c) for c in g.columns]
    reduce_memory(g)

    return g


def main():
    res = []
    with ProcessPoolExecutor() as executor:
        res = executor.map(_tail, [12, 24])
    res = pd.concat(res, axis=1).reset_index()
    res.to_feather('./data/credit.tail.feather')


if __name__ == '__main__':
    main()
