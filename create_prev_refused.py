import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)


def _create():
    df = pd.read_feather('./data/prev.preprocessed.feather')
    df, cat_columns = one_hot_encoder(df)
    df = df[df['FLAG_Refused'] == 1]

    a = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'AMT_APPLICATION': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'AMT_CREDIT': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'RATIO_APP_TO_CREDIT': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'AMT_DOWN_PAYMENT': ['sum', 'median'],
        'AMT_GOODS_PRICE': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'HOUR_APPR_PROCESS_START': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'DAYS_DECISION': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'CNT_PAYMENT': ['sum', 'median', 'mean', 'std', 'max', 'min'],
        'RATE_DOWN_PAYMENT': ['sum', 'median', 'mean', 'std', 'max', 'min'],
    }

    g = df.groupby('SK_ID_CURR').agg(a)
    g.columns = ['PREV_REFUSED_{}_{}'.format(a, b.upper()) for a, b in g.columns]

    return g.reset_index()


def main():
    agg = _create()
    reduce_memory(agg)
    agg.to_feather('./data/prev.refused.feather')


if __name__ == '__main__':
    main()
