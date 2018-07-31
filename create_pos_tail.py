import pandas as pd
from utility import reduce_memory
from concurrent.futures import ProcessPoolExecutor


def _tail(k):
    print('create pos tail {}...'.format(k))
    df = pd.read_feather('./data/pos.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df = df.groupby('SK_ID_CURR').tail(k).reset_index(drop=True)
    grp = df.groupby('SK_ID_CURR')

    g = grp.agg({
        'SK_DPD': ['sum', 'mean', 'std', 'max', 'skew'],
        'SK_DPD_DEF': ['sum', 'mean', 'std', 'max', 'skew'],
        'FLAG_LATE': ['sum', 'mean'],
        'FLAG_LATE_DEF': ['sum', 'mean'],
    })
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g.columns = ['POS_TAIL_{}_{}'.format(k, c) for c in g.columns]
    reduce_memory(g)

    return g


def main():
    res = []
    with ProcessPoolExecutor() as executor:
        res = executor.map(_tail, [6, 12, 30])
    res = pd.concat(res, axis=1).reset_index()
    res.to_feather('./data/pos.tail.feather')


if __name__ == '__main__':
    main()
