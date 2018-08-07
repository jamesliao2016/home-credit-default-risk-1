import pandas as pd
from utility import reduce_memory


def _app():
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    test = pd.read_feather('./data/application_test.preprocessed.feather')

    return pd.concat([train, test], sort=False)


def _create(by, fs):
    print('create {}...'.format(by))
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    app = _app()
    df = df.merge(app, on='SK_ID_CURR', how='left')
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
        # 'EXT_SOURCE_1': ['mean'],
        # 'EXT_SOURCE_2': ['mean'],
        # 'EXT_SOURCE_3': ['mean'],
        'EXT_SOURCES_MEAN': ['mean'],
        'ANNUITY_LENGTH': ['mean'],
    }

    for by in [
        ['CREDIT_CURRENCY'],
        ['CREDIT_TYPE'],
        # ['CREDIT_ACTIVE'],
    ]:
        res.append(_create(by, fs))

    res = pd.concat(res, axis=1)
    res.columns = ['BURE_{}'.format(c) for c in res.columns]
    res = res.reset_index()

    print(res.columns, res.columns.shape)
    res.to_feather('./data/bureau.grp.feather')


if __name__ == '__main__':
    main()
