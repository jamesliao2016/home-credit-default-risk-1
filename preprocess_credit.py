import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)


def create_diff(cred_df):
    ts_features = [
        'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_INST_MIN_REGULARITY',
        'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL',
        'AMT_TOTAL_RECEIVABLE',
        'CNT_INSTALMENT_MATURE_CUM',
        'SK_DPD',
        'SK_DPD_DEF',
    ]
    grp = cred_df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    diff_df = grp[ts_features].diff()
    diff_df.columns = ['DIFF_{}'.format(c) for c in diff_df.columns]
    cred_df = pd.concat([cred_df, diff_df], axis=1)
    return cred_df


def preprocess_credit():
    cred_df = pd.read_feather('./data/credit_card_balance.feather')
    cred_df = cred_df.sort_values(
        ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
    cred_df = create_diff(cred_df)

    cred_df['SK_DPD_PLUS'] = (cred_df['SK_DPD'] > 0).astype('i')
    cred_df['SHORT_PAYMENT'] = (
        cred_df['AMT_INST_MIN_REGULARITY'] >
        cred_df['AMT_PAYMENT_TOTAL_CURRENT']
    ).astype('i')

    cred_df['RATIO_USED_LIMIT'] = cred_df[
        'AMT_DRAWINGS_CURRENT'] / cred_df['AMT_CREDIT_LIMIT_ACTUAL']

    cred_df['RATIO_PAYED'] = cred_df[
        'AMT_PAYMENT_TOTAL_CURRENT'] / cred_df['AMT_INST_MIN_REGULARITY']
    cred_df['RATIO_PAYED'].fillna(1, inplace=True)
    cred_df['FAIL_PAY_TO_PRINCIPAL'] = (
        (cred_df['RATIO_PAYED'] < 1) &
        (cred_df['AMT_RECEIVABLE_PRINCIPAL'] > 0)
    ).astype('i')
    cred_df['FAIL_PAY_TO_TOTAL'] = (
        (cred_df['RATIO_PAYED'] < 1) &
        (cred_df['AMT_TOTAL_RECEIVABLE'] > 0)
    ).astype('i')
    cred_df['DIFF_PAYMENT_TOTAL_AND_PLAIN'] = cred_df[
        'AMT_PAYMENT_TOTAL_CURRENT'] - cred_df['AMT_PAYMENT_CURRENT']
    cred_df['DIFF_RECEIVABLE_TOTAL_AND_PLAIN'] = cred_df[
        'AMT_TOTAL_RECEIVABLE'] - cred_df['AMT_RECIVABLE']
    cred_df['DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL'] = cred_df[
        'AMT_TOTAL_RECEIVABLE'] - cred_df['AMT_RECEIVABLE_PRINCIPAL']

    return cred_df


def main():
    cred_df = preprocess_credit()
    cred_df.to_feather('./data/credit_card_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
