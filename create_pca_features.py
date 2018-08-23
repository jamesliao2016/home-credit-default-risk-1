import pandas as pd
import fbpca


def main():
    X = pd.read_feather('./data/features.normalized.feather')
    idx = X.pop('SK_ID_CURR')
    X.pop('TARGET')
    for c in X.columns:
        if str(X[c].dtype) == 'category':
            X.pop(c)

    dtypes = {}
    for c in X.columns:
        dtypes[c] = 'float32'
    X = X.astype(dtypes)
    X = X.values
    print('decompose...')
    U, s, Va = fbpca.pca(X, k=64)
    print(U.shape)
    print(s)

    del X
    res = pd.DataFrame(U)
    res.columns = ['pca_{}'.format(i) for i in range(U.shape[1])]
    res['SK_ID_CURR'] = idx

    res.to_feather('./data/pca.feather')


if __name__ == '__main__':
    main()
