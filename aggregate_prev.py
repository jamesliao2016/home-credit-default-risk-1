import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)


def preprocess_prev():
    df = pd.read_feather('./data/prev.preprocessed.feather')
    df, cat_columns = one_hot_encoder(df)

    # Previous applications numeric features
    fs = ['sum', 'median', 'mean', 'std', 'max', 'min']
    a = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': fs,
        'AMT_APPLICATION': fs,
        'AMT_CREDIT': fs,
        'RATIO_APP_TO_CREDIT': fs,
        'AMT_DOWN_PAYMENT': fs,
        'AMT_GOODS_PRICE': fs,
        'HOUR_APPR_PROCESS_START': fs,
        'RATE_DOWN_PAYMENT': fs,
        'DAYS_DECISION': fs,
        'CNT_PAYMENT': fs,
        # added
        'FLAG_Refused': ['mean'],
        'NOT_COMPLETE': ['mean'],
        'FLAG_X_SELL_1': ['mean'],
        'FLAG_X_SELL_2': ['mean'],
    }

    grp = df.groupby('SK_ID_CURR')
    g = grp.agg(a)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g['COUNT'] = grp.size()
    g.columns = ['PREV_{}'.format(c) for c in g.columns]

    return g.reset_index()


def main():
    agg = preprocess_prev()
    reduce_memory(agg)
    agg.to_feather('./data/prev.agg.feather')


if __name__ == '__main__':
    main()
