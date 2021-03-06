import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    df = pd.read_feather('./data/inst.preprocessed.feather')
    df, _ = one_hot_encoder(df)
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])

    df = df.drop(['IS_CREDIT'], axis=1)

    a = {
        'AMT_PAYMENT': ['mean', 'min', 'max', 'std'],
        'AMT_INSTALMENT': ['mean', 'min', 'max', 'std'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'min', 'max', 'std'],
        'NUM_INSTALMENT_VERSION': ['mean'],
        'NUM_INSTALMENT_NUMBER': ['mean', 'min', 'max', 'std'],
        'DBD': ['mean', 'min', 'max', 'std'],
        'DPD': ['mean', 'min', 'max', 'std'],
        'FLAG_DBD': ['mean'],
        'FLAG_DPD': ['mean'],
        'DAYS_INSTALMENT': ['mean'],
        'RATIO_PAYMENT': ['mean'],
        'DIFF_PAYMENT': ['mean'],
    }

    grp = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    prev = grp.shift(-1)
    for c in prev.columns:
        if df[c].dtype == 'object':
            continue
        df[c] -= prev[c]

    grp = df.groupby('SK_ID_CURR')
    agg = grp.agg(a)
    agg.columns = ["{}_{}".format(a, b.upper()) for a, b in agg.columns]
    agg.columns = ["INST_DIFF_{}".format(c) for c in agg.columns]
    agg = agg.reset_index()

    reduce_memory(agg)

    agg.to_feather('./data/inst.diff.feather')


if __name__ == '__main__':
    main()
