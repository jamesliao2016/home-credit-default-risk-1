import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression


def _func(y):
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression(n_jobs=-1)
        lr.fit(x, y)
        trend = lr.coef_[0]
    except Exception as e:
        trend = np.nan

    return trend


def f(g):
    return g.apply(_func)


def _trend(k):
    print('create credit trend {}...'.format(k))
    df = pd.read_feather('./data/credit.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df = df.groupby('SK_ID_CURR').tail(k)
    grp = df.groupby('SK_ID_CURR')

    columns = [
        'AMT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_ATM_CURRENT',
        'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_INST_MIN_REGULARITY',
        'SK_DPD',
        # added
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL',
        'DIFF_PAYMENT_TOTAL_AND_PLAIN',
    ]
    data = [grp[c] for c in columns]
    res = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'].unique()})
    res = res.set_index('SK_ID_CURR')
    with ProcessPoolExecutor() as executor:
        agg = executor.map(f, data)

    for c, agg in zip(columns, agg):
        res[c] = agg

    res.columns = ['CRED_TREND_{}_{}'.format(k, c) for c in res.columns]

    return res


def main():
    res = []
    with ProcessPoolExecutor() as executor:
        res = executor.map(_trend, [6, 12])
    res = pd.concat(res, axis=1).reset_index()
    res.to_feather('./data/credit.trend.feather')


if __name__ == '__main__':
    main()
