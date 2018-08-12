import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def _aggregate():
    df = pd.read_feather('./data/credit.preprocessed.feather')

    grp = df.groupby('SK_ID_CURR')

    fs = ['mean', 'std', 'min', 'max', 'sum']
    a = {
        'RATIO_USED_LIMIT': ['mean', 'std', 'max', 'sum'],
        'SK_DPD_PLUS': ['mean', 'std'],
        'SHORT_PAYMENT': ['std', 'sum'],
        'AMT_BALANCE': ['mean', 'std', 'min'],
        'AMT_CREDIT_LIMIT_ACTUAL': fs,
        'AMT_DRAWINGS_CURRENT': ['std', 'min', 'max', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'std', 'max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['mean', 'std', 'max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max'],
        'AMT_PAYMENT_CURRENT': fs,
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'std', 'max', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['std', 'min', 'max'],
        'AMT_RECIVABLE': ['mean', 'std', 'min', 'max'],
        'AMT_TOTAL_RECEIVABLE': ['std', 'min', 'max', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'std', 'max', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['mean', 'std', 'max', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': fs,
        'CNT_INSTALMENT_MATURE_CUM': ['std', 'min', 'max', 'sum'],
        'SK_DPD': ['mean', 'std', 'max'],
        'SK_DPD_DEF': ['mean', 'std', 'max', 'sum'],
        'RATIO_PAYED': fs,
        'FAIL_PAY_TO_PRINCIPAL': ['mean', 'std', 'min', 'max'],
        'FAIL_PAY_TO_TOTAL': ['mean', 'std', 'max'],
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': fs,
        'DIFF_RECEIVABLE_TOTAL_AND_PLAIN': ['mean'],
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': fs,
    }
    g = grp.agg(a)
    g.columns = ['{}_{}'.format(x, y.upper()) for x, y in g.columns]
    g['COUNT'] = grp.size()
    agg = g

    # another features
    agg['MAX_RATIO_INSTALMENT'] = agg['CNT_INSTALMENT_MATURE_CUM_MAX'] / agg['COUNT']

    agg.columns = ['CRED_{}'.format(c) for c in agg.columns]

    # TODO: diff by max-min
    # TODO: handle NAME_CONTRACT_STATUS

    return agg.reset_index()


def main():
    agg = _aggregate()
    agg.to_feather('./data/credit.agg.feather')


if __name__ == '__main__':
    main()
