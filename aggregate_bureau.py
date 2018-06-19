import pandas as pd
# from utility import one_hot_encoder
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def handle_categorical(df):
    cat_columns = [
        'CREDIT_ACTIVE',
        'CREDIT_CURRENCY',
        'CREDIT_TYPE',
        'BB_LAST_STATUS',
    ]
    df = df[['SK_ID_CURR'] + cat_columns]
    df = pd.get_dummies(df, columns=cat_columns, dummy_na=True)
    df.columns = [str(c).replace(' ', '_') for c in df.columns]
    agg = df.groupby('SK_ID_CURR').mean()
    return agg


def aggregate_bureau():
    df = pd.read_feather('./data/bureau.preprocessed.feather')

    cat_agg = handle_categorical(df)

    a = {
        # original
        'CREDIT_ACTIVE': ['nunique'],
        'CREDIT_CURRENCY': ['nunique'],
        'DAYS_CREDIT': ['mean', 'min', 'max', 'std'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max', 'std'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'max'],
        'DAYS_ENDDATE_FACT': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],
        'AMT_CREDIT_SUM_LIMIT': ['sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum'],
        'CREDIT_TYPE': ['nunique'],
        'DAYS_CREDIT_UPDATE': ['mean', 'max'],
        'AMT_ANNUITY': ['sum'],
        # preprocessed
        'DAYS_CREDIT_ENDDATE_PLUS': ['sum', 'mean'],
        'TSDIFF_DAYS_CREDIT': ['mean', 'std', 'min', 'max'],
        'TSDIFF_DAYS_CREDIT_ENDDATE': ['mean', 'std', 'min', 'max'],
        'TSDIFF_DAYS_ENDDATE_FACT': ['mean', 'std', 'min', 'max'],
        'TSDIFF_DAYS_CREDIT_UPDATE': ['mean', 'std', 'min', 'max'],
        'DIFF_ENDDATE': ['mean', 'std', 'min', 'max'],
        'GOOD_BB_LAST_STATUS': ['mean'],
        'BAD_BB_LAST_STATUS': ['mean'],
        # bb
        'BB_FIRST_MONTHS_BALANCE': ['mean', 'min'],
        'BB_LAST_MONTHS_BALANCE': ['mean', 'max'],
        'BB_LAST_STATUS': ['nunique'],
        'BB_COUNT': ['sum', 'mean'],
        'BB_STATUS_NUNIQUE': ['mean'],
        'BB_GOOD_STATUS_MEAN': ['mean'],
        'BB_BAD_STATUS_MEAN': ['mean'],
        'BB_TERM': ['sum', 'mean'],
    }

    grp = df.groupby('SK_ID_CURR')
    g = grp.agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    agg = g

    agg['RATIO_CREDIT_DEBT'] = agg[
        'AMT_CREDIT_SUM_DEBT_SUM'] / agg['AMT_CREDIT_SUM_SUM']
    agg['RATIO_CREDIT_OVERDUE'] = agg[
        'AMT_CREDIT_SUM_OVERDUE_SUM'] / agg['AMT_CREDIT_SUM_DEBT_SUM']

    # Bureau: Active credits - using only numerical aggregations
    act = df[df['CREDIT_ACTIVE'] == 'Active']
    g = act.groupby('SK_ID_CURR').agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['ACT_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    # # Bureau: Closed credits - using only numerical aggregations
    clo = df[df['CREDIT_ACTIVE'] == 'Closed']
    g = clo.groupby('SK_ID_CURR').agg(a)
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['CLO_{}'.format(c) for c in g.columns]
    agg = agg.join(g, on='SK_ID_CURR', how='left')

    agg = agg.join(cat_agg, on='SK_ID_CURR', how='left')
    agg.columns = ['BURE_{}'.format(c) for c in agg.columns]

    agg['BURE_ACT_AMT_CREDIT_SUM_SUM'].fillna(0, inplace=True)
    agg['BURE_ACT_AMT_ANNUITY_SUM'].fillna(0, inplace=True)

    return agg.reset_index()


def main():
    agg = aggregate_bureau()
    agg.to_feather('./data/bureau.agg.feather')


if __name__ == '__main__':
    main()
