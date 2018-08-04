import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def _aggregate():
    df = pd.read_feather('./data/credit.preprocessed.feather')

    grp = df.groupby('SK_ID_CURR')

    fs = ['mean', 'std', 'min', 'max', 'sum']
    a = {
        'RATIO_USED_LIMIT': fs,
        'SK_DPD_PLUS': fs,
        'SHORT_PAYMENT': fs,
        'AMT_BALANCE': fs,
        'AMT_CREDIT_LIMIT_ACTUAL': fs,
        'AMT_DRAWINGS_CURRENT': fs,
        'AMT_DRAWINGS_ATM_CURRENT': fs,
        'AMT_DRAWINGS_POS_CURRENT': fs,
        'AMT_DRAWINGS_OTHER_CURRENT': fs,
        'AMT_INST_MIN_REGULARITY': fs,
        'AMT_PAYMENT_CURRENT': fs,
        'AMT_PAYMENT_TOTAL_CURRENT': fs,
        'AMT_RECEIVABLE_PRINCIPAL': fs,
        'AMT_RECIVABLE': fs,
        'AMT_TOTAL_RECEIVABLE': fs,
        'CNT_DRAWINGS_ATM_CURRENT': fs,
        'CNT_DRAWINGS_CURRENT': fs,
        'CNT_DRAWINGS_OTHER_CURRENT': fs,
        'CNT_DRAWINGS_POS_CURRENT': fs,
        'CNT_INSTALMENT_MATURE_CUM': fs,
        'SK_DPD': fs,
        'SK_DPD_DEF': fs,
        'RATIO_PAYED': fs,
        'FAIL_PAY_TO_PRINCIPAL': fs,
        'FAIL_PAY_TO_TOTAL': fs,
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
