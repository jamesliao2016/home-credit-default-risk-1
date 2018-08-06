import pandas as pd
from utility import reduce_memory


def _create(by, fs):
    print('create {}...'.format(by))
    df = pd.read_feather('./data/pos.preprocessed.feather')
    agg = df.groupby(by).agg(fs)
    agg.columns = ['GRP_{}_{}_{}'.format('_'.join(by), a, b.upper()) for a, b in agg.columns]
    features = []
    df = df.set_index(by)
    df = df.join(agg, on=by, how='left')
    df = df.reset_index(drop=True)
    for c, ms in fs.items():
        for m in ms:
            ac = 'GRP_{}_{}_{}'.format('_'.join(by), c, m.upper())
            dc = 'DIFF_{}'.format(ac)
            adc = 'ABS_DIFF_{}'.format(ac)
            df[dc] = df[c] - df[ac]
            df[adc] = (df[c] - df[ac]).abs()
            features += [dc, adc]
    df = df[['SK_ID_CURR'] + features]
    df = df.groupby('SK_ID_CURR').mean()
    reduce_memory(df)
    return df


def main():
    res = []

    fs = {
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum', 'max', 'min', 'std'],
    }

    for by in [
        ['NAME_CONTRACT_STATUS'],
    ]:
        res.append(_create(by, fs))

    res = pd.concat(res, axis=1)
    res.columns = ['POS_{}'.format(c) for c in res.columns]
    res = res.reset_index()

    print(res.columns, res.columns.shape)
    res.to_feather('./data/pos.grp.feather')


if __name__ == '__main__':
    main()
