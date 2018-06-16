import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def add_diff(df):
    ts_features = [
        'DAYS_CREDIT',
        'DAYS_CREDIT_ENDDATE',
        'DAYS_ENDDATE_FACT',
        'DAYS_CREDIT_UPDATE',
    ]
    grp = df.groupby('SK_ID_CURR')
    diff = grp.diff()[ts_features]
    diff.columns = ['TSDIFF_{}'.format(c) for c in diff.columns]
    for c in diff.columns:
        df[c] = diff[c]


def preprocess_bureau():
    df = join_bb()
    df = df.sort_values(
        ['SK_ID_CURR', 'DAYS_CREDIT'],
    ).reset_index(drop=True)
    df = df[:10000]
    df['DAYS_CREDIT_ENDDATE_PLUS'] = (
        df['DAYS_CREDIT_ENDDATE'] >= 0).astype('i')
    df['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    df['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    df['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    df['AMT_CREDIT_SUM_LIMIT'].fillna(0, inplace=True)
    df['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    df['AMT_ANNUITY'].fillna(0, inplace=True)
    df['BB_COUNT'].fillna(0, inplace=True)
    df['BB_STATUS_NUNIQUE'].fillna(0, inplace=True)
    df['BB_GOOD_STATUS_MEAN'].fillna(1, inplace=True)
    df['BB_BAD_STATUS_MEAN'].fillna(0, inplace=True)
    df['BB_TERM'].fillna(0, inplace=True)
    add_diff(df)
    return df


def join_bb():
    df = pd.read_feather('./data/bureau.feather')
    agg = pd.read_feather('./data/bureau_balance.agg.feather')
    df = df.merge(agg, on='SK_ID_BUREAU', how='left')
    del df['SK_ID_BUREAU']
    return df


def main():
    bure_df = preprocess_bureau()
    bure_df.to_feather('data/bureau.preprocessed.feather')


if __name__ == '__main__':
    main()
