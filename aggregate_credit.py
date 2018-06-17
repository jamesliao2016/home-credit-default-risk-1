

def aggregate_credit(df, key):
    grp = df.groupby(key)

    g = grp[['MONTHS_BALANCE']].count()
    g.columns = ['COUNT']
    agg = g

    fs = ['mean', 'std', 'min', 'max', 'nunique']
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
        'DIFF_AMT_BALANCE': fs,
        'DIFF_AMT_CREDIT_LIMIT_ACTUAL': fs,
        'DIFF_AMT_INST_MIN_REGULARITY': fs,
        'DIFF_AMT_PAYMENT_TOTAL_CURRENT': fs,
        'DIFF_AMT_RECEIVABLE_PRINCIPAL': fs,
        'DIFF_AMT_TOTAL_RECEIVABLE': fs,
        'DIFF_CNT_INSTALMENT_MATURE_CUM': fs,
        'DIFF_SK_DPD': fs,
        'DIFF_SK_DPD_DEF': fs,
        'RATIO_PAYED': fs,
        'FAIL_PAY_TO_PRINCIPAL': fs,
        'FAIL_PAY_TO_TOTAL': fs,
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': fs,
        'DIFF_RECEIVABLE_TOTAL_AND_PLAIN': fs,
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': fs,
    }
    g = grp.agg(a)
    g.columns = ['{}_{}'.format(x, y.upper()) for x, y in g.columns]
    agg = agg.join(g, on=key, how='left')

    # another features
    agg['MAX_RATIO_INSTALMENT'] =\
        agg['CNT_INSTALMENT_MATURE_CUM_MAX'] / agg['COUNT']

    # TODO: diff by max-min
    # TODO: handle NAME_CONTRACT_STATUS

    return agg
