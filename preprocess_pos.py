import pandas as pd
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 300)


def create_diff(df):
    ts_features = [
        'CNT_INSTALMENT',
        'CNT_INSTALMENT_FUTURE',
        'SK_DPD',
        'SK_DPD_DEF',
    ]
    grp = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    diff_df = grp[ts_features].diff()
    diff_df.columns = ['DIFF_{}'.format(c) for c in diff_df.columns]
    df = pd.concat([df, diff_df], axis=1)
    return df


def preprocess_pos():
    df = pd.read_feather('./data/POS_CASH_balance.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
    df = create_diff(df)
    return df.reset_index(drop=True)


def main():
    df = preprocess_pos()
    df.to_feather('./data/POS_CASH_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
