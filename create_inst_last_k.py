import pandas as pd
from utility import reduce_memory


def last_k(k):
    print('create inst last {}...'.format(k))
    ins = pd.read_feather('./data/inst.preprocessed.feather')
    ins.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'], inplace=True)
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

    return g.reset_index()


def main():
    for k in [1, 5, 10, 20, 50, 100]:
        res = last_k(k)
        res.to_feather('./data/inst.last.{}.feather'.format(k))


if __name__ == '__main__':
    main()
