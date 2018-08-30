import pandas as pd
from preprocess_bb import status_to_dpd
from utility import reduce_memory


def _create():
    df = pd.read_feather('./data/bureau_balance.feather')
    df = df[df['STATUS'] != 'C']
    df = df.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])
    df = df.groupby('SK_ID_BUREAU').last()

    bure = pd.read_feather('./data/bureau.feather')
    df = df.merge(bure[['SK_ID_BUREAU', 'SK_ID_CURR']], on='SK_ID_BUREAU', how='left')
    df = df.reset_index(drop=True)
    df['DPD'] = df['STATUS'].apply(status_to_dpd)

    g = df.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'std'],
        'DPD': ['min', 'max', 'mean', 'std'],
    })
    g.columns = ['{}_{}'.format(x, y.upper()) for x, y in g.columns]
    g.columns = ['BB_OPEN_{}'.format(c) for c in g.columns]
    reduce_memory(g)

    g = g.reset_index()

    return g


def main():
    agg = _create()
    agg.to_feather('./data/bb.open.feather')


if __name__ == '__main__':
    main()
