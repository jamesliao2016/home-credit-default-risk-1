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

    df.drop(['SK_ID_BUREAU'], axis=1)
    add_diff(df)
    return df


def load_bure():
    bure = pd.read_feather('./data/bureau.feather')
    bure['FINISHED'] = (bure['DAYS_ENDDATE_FACT'] <= 0).astype('i')
    indexer = pd.isnull(bure['DAYS_CREDIT_ENDDATE'])
    bure.loc[indexer, 'DAYS_CREDIT_ENDDATE'] = bure.loc[indexer, 'DAYS_ENDDATE_FACT']
    bure['DIFF_ENDDATE'] = bure['DAYS_ENDDATE_FACT'] - bure['DAYS_CREDIT_ENDDATE']
    bure['TERM'] = bure['DAYS_CREDIT_ENDDATE'] - bure['DAYS_CREDIT']
    bure['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    bure['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    bure['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    bure['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    bure['AMT_ANNUITY'].fillna(0, inplace=True)
    return bure


def join_bb():
    bure = load_bure()
    bb = pd.read_feather('./data/bb.agg.feather')
    bure = bure.merge(bb, on='SK_ID_BUREAU', how='left')
    return bure


def main():
    bure = preprocess_bureau()
    bure.to_feather('data/bureau.preprocessed.feather')


if __name__ == '__main__':
    main()
