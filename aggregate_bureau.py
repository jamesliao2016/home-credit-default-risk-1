import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def aggregate_bureau():
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    grp = df.groupby('SK_ID_CURR')

    # agg
    agg = {
        # cat
        'CREDIT_TYPE': ['nunique'],
        # num
        'AMT_ANNUITY': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'std', 'sum', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT_UPDATE': ['mean', 'std', 'sum', 'min', 'max'],
        # new
        'FLAG_ACTIVE': ['mean'],
        'FLAG_ONGOING': ['mean'],
        # bb
        "BB_DPD_MEAN": ['mean', 'std', 'sum', 'min', 'max'],
        "BB_STATUS_NUNIQUE": ['std', 'sum', 'min', 'max'],
    }

    g = grp.agg(agg)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g['COUNT'] = grp.size()
    f = g

    # discussion/57175
    f['RATIO_COUNT_PER_TYPE'] = f['COUNT'] / f['CREDIT_TYPE_NUNIQUE']

    f['RATIO_DEBT_PER_CREDIT'] = f['AMT_CREDIT_SUM_DEBT_SUM'] / f['AMT_CREDIT_SUM_SUM']

    f['RATIO_OVERDUE_PER_DEBT'] = f['AMT_CREDIT_SUM_OVERDUE_SUM'] / f['AMT_CREDIT_SUM_DEBT_SUM']

    # # # Bureau: Closed credits
    # clo = df[df['FLAG_ACTIVE'] == 0]
    # g = clo.groupby('SK_ID_CURR').agg(agg)
    # g.columns = [x + "_" + y.upper() for x, y in g.columns]
    # g.columns = ['CLO_{}'.format(c) for c in g.columns]
    # f = f.join(g, on='SK_ID_CURR', how='left')

    f.columns = ['BURE_{}'.format(c) for c in f.columns]

    return f.reset_index()


def main():
    agg = aggregate_bureau()
    agg.to_feather('./data/bureau.agg.feather')


if __name__ == '__main__':
    main()
