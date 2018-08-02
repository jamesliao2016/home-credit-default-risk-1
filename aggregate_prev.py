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
    }
    for c in cat_columns:
        a[c] = ['mean']

    for c in df.columns:
        if c == 'SK_ID_CURR' or c == 'SK_ID_PREV':
            continue
        if c in a:
            continue
        a[c] = ['mean']

    grp = df.groupby('SK_ID_CURR')
    g = grp.agg(a)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    g['COUNT'] = grp.size()
    agg = g

    # Previous Applications: Approved Applications
    approved = df[df['FLAG_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(a)
    approved_agg.columns = [
        'APPROVED_{}_{}'.format(
                a.upper(), b) for a, b in approved_agg.columns]

    agg = agg.join(approved_agg, on='SK_ID_CURR', how='left')

    # Previous Applications: Refused Applications
    refused = df[df['FLAG_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(a)
    refused_agg.columns = [
        'REFUSED_{}_{}'.format(a.upper(), b) for a, b in refused_agg.columns]
    agg = agg.join(refused_agg, on='SK_ID_CURR', how='left')

    agg.columns = ['PREV_{}'.format(c) for c in agg.columns]

    return agg.reset_index()


def main():
    agg = preprocess_prev()
    reduce_memory(agg)
    agg.to_feather('./data/prev.agg.feather')


if __name__ == '__main__':
    main()
