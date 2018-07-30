import pandas as pd
from utility import reduce_memory
from concurrent.futures import ProcessPoolExecutor


def last_k(k):
    print('create inst tail {}...'.format(k))
    ins = pd.read_feather('./data/inst.preprocessed.feather')
    ins = ins.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])
    ins = ins.groupby('SK_ID_CURR').tail(k).reset_index(drop=True)
    grp = ins.groupby('SK_ID_CURR')

    agg = {
        'NUM_INSTALMENT_VERSION': [
            'sum', 'mean', 'max', 'min', 'std', 'median', 'skew'],
        'DPD': ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew'],
        'FLAG_DPD': ['sum', 'mean'],
        'DIFF_PAYMENT': ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew'],
        'FLAG_DIFF_PAYMENT': ['sum', 'mean'],
    }
    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g.columns = ['INST_LAST_{}_{}'.format(k, c) for c in g.columns]
    reduce_memory(g)

    return g


def main():
    res = []
    with ProcessPoolExecutor() as executor:
        res = executor.map(last_k, [1, 5, 10, 20, 50])
    res = pd.concat(res, axis=1).reset_index()
    res.to_feather('./data/inst.tail.feather')


if __name__ == '__main__':
    main()
