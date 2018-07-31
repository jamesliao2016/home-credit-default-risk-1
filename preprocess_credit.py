import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", 200)


def preprocess_credit():
    cred_df = pd.read_feather('./data/credit_card_balance.feather')

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
    cred_df.to_feather('./data/credit.preprocessed.feather')


if __name__ == '__main__':
    main()
