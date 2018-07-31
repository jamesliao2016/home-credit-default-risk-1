import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", 200)


def preprocess_credit():
    df = pd.read_feather('./data/credit_card_balance.feather')

    df['SK_DPD_PLUS'] = (df['SK_DPD'] > 0).astype('i')
    df['SHORT_PAYMENT'] = (
        df['AMT_INST_MIN_REGULARITY'] >
        df['AMT_PAYMENT_TOTAL_CURRENT']
    ).astype('i')

    df['RATIO_USED_LIMIT'] = df['AMT_DRAWINGS_CURRENT'] / df['AMT_CREDIT_LIMIT_ACTUAL']

    df['RATIO_PAYED'] = df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_INST_MIN_REGULARITY']
    df['RATIO_PAYED'].fillna(1, inplace=True)
    df['FAIL_PAY_TO_PRINCIPAL'] = (
        (df['RATIO_PAYED'] < 1) &
        (df['AMT_RECEIVABLE_PRINCIPAL'] > 0)
    ).astype('i')
    df['FAIL_PAY_TO_TOTAL'] = (
        (df['RATIO_PAYED'] < 1) &
        (df['AMT_TOTAL_RECEIVABLE'] > 0)
    ).astype('i')
    df['DIFF_PAYMENT_TOTAL_AND_PLAIN'] = df['AMT_PAYMENT_TOTAL_CURRENT'] - df['AMT_PAYMENT_CURRENT']
    df['DIFF_RECEIVABLE_TOTAL_AND_PLAIN'] = df['AMT_TOTAL_RECEIVABLE'] - df['AMT_RECIVABLE']
    df['DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL'] =\
        df['AMT_TOTAL_RECEIVABLE'] - df['AMT_RECEIVABLE_PRINCIPAL']

    return df


def main():
    df = preprocess_credit()
    df.to_feather('./data/credit.preprocessed.feather')


if __name__ == '__main__':
    main()
