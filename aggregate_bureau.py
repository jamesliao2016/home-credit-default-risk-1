import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def aggregate_bureau():
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    columns = [c for c in df.columns if df[c].dtype != 'object']
    df = df[columns]

    a = {}
    for c in columns:
        if c == 'SK_ID_CURR':
            continue
        if df[c].dtype == 'int64':
            a[c] = ['mean', 'std', 'sum', 'min', 'max', 'nunique']
            df[c] = df[c].astype('int32')
        else:
            a[c] = ['mean', 'std', 'sum', 'min', 'max']
            df[c] = df[c].astype('float32')

    grp = df.groupby('SK_ID_CURR')
    g = grp.agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    agg = g

    # Bureau: Active credits
    act = df[df['FINISHED'] == 0]
    g = act.groupby('SK_ID_CURR').agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['ACT_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    # # Bureau: Closed credits
    clo = df[df['FINISHED'] == 1]
    g = clo.groupby('SK_ID_CURR').agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['CLO_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    agg.columns = ['BURE_{}'.format(c) for c in agg.columns]

    agg['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)
    agg['BURE_ACT_AMT_ANNUITY_SUM'].fillna(0, inplace=True)

    return agg.reset_index()


def main():
    agg = aggregate_bureau()
    agg.to_feather('./data/bureau.agg.num.feather')


if __name__ == '__main__':
    main()
