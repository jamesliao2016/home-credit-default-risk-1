import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def filter_last_k(k):
    train = pd.read_feather('./data/application_train.feather')
    train = train[['SK_ID_CURR', 'TARGET']]
    ins = pd.read_feather('./data/inst.last.{}.feather'.format(k))
    train = train.merge(ins, on='SK_ID_CURR')
    train.drop(['SK_ID_CURR'], 1)
    corr = train.corr()[['TARGET']]
    corr['TARGET'] = corr['TARGET'].apply(abs)
    corr = corr.sort_values('TARGET', ascending=False)
    print(corr)

    cols = corr[corr['TARGET'] > 0.01].index.values.tolist()
    cols.remove('TARGET')
    if 'SK_ID_CURR' not in cols:
        cols.append('SK_ID_CURR')
    return ins[cols].set_index('SK_ID_CURR')


def main():
    res = []
    for k in [1, 5, 10, 20, 50, 100]:
        res.append(filter_last_k(k))
    res = pd.concat(res, axis=1)
    print(res.head())
    res = res.reset_index()
    res.to_feather('./data/ins.last.k.feather')


if __name__ == '__main__':
    main()
