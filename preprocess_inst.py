import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def add_diff(df):
    ts_features = [
        'DAYS_INSTALMENT',
        'DAYS_ENTRY_PAYMENT',
        'AMT_INSTALMENT',
        'AMT_PAYMENT',
    ]
    g = df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    g = g[ts_features].diff()
    for c in ts_features:
        df['TSDIFF_{}'.format(c)] = g[c]


def preprocess_inst():
    ins_df = pd.read_feather('./data/installments_payments.feather')
    ins_df = ins_df.sort_values(
        ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])
    add_diff(ins_df)

    # Percentage and difference paid in each installment (amount paid and installment value) # noqa
    ins_df['RATIO_PAYMENT'] = ins_df['AMT_PAYMENT'] / ins_df['AMT_INSTALMENT']
    ins_df['DIFF_PAYMENT'] = ins_df['AMT_INSTALMENT'] - ins_df['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins_df['DPD'] = ins_df['DAYS_ENTRY_PAYMENT'] - ins_df['DAYS_INSTALMENT']
    ins_df['DBD'] = ins_df['DAYS_INSTALMENT'] - ins_df['DAYS_ENTRY_PAYMENT']
    ins_df['DPD'] = ins_df['DPD'].apply(lambda x: x if x > 0 else 0)
    ins_df['DBD'] = ins_df['DBD'].apply(lambda x: x if x > 0 else 0)

    return ins_df.reset_index(drop=True)


def main():
    res = preprocess_inst()
    res.to_feather('./data/installments_payments.preprocessed.feather')


if __name__ == '__main__':
    main()
