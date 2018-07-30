import pandas as pd
from utility import reduce_memory


def _create():
    df = pd.read_feather('./data/pos.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
    grp = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])

    nex = grp.shift(1)
    for c in nex.columns:
        if df[c].dtype == 'object':
            continue
        df[c] -= nex[c]

    grp = df.groupby('SK_ID_CURR')
    g = grp.agg({
        'CNT_INSTALMENT': ['mean', 'std', 'max', 'min'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'std', 'max', 'min'],
        'SK_DPD': ['min', 'max'],
        'SK_DPD_DEF': ['min', 'max'],
    })
    g.columns = ['{}_{}'.format(a, b.upper()) for a, b in g.columns]
    g.columns = ['POS_DIFF_{}'.format(c) for c in g.columns]

    return g.reset_index()


def main():
    df = _create()
    reduce_memory(df)
    df.to_feather('./data/pos.diff.feather')


if __name__ == '__main__':
    main()
