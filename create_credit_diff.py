import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    df = pd.read_feather('./data/credit.preprocessed.feather')

    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    grp = df.groupby(['SK_ID_CURR'])
    prev = grp.shift(-1)
    for c in prev.columns:
        if df[c].dtype == 'object':
            continue
        df[c] -= prev[c]

    grp = df.groupby('SK_ID_CURR')
    agg = grp.agg({
        'AMT_BALANCE': ['mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'],
        'SK_DPD': ['mean', 'sum'],
        # added
        'FAIL_PAY_TO_TOTAL': ['mean'],
        'FAIL_PAY_TO_PRINCIPAL': ['mean'],
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': ['mean', 'sum'],
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': ['mean', 'sum'],
    })
    agg.columns = [a + "_" + b.upper() for a, b in agg.columns]
    agg.columns = ["CRED_DIFF_{}".format(c) for c in agg.columns]
    agg = agg.reset_index()

    agg.to_feather('./data/credit.diff.feather')


if __name__ == '__main__':
    main()
