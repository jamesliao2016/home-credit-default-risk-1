

def aggregate_credit(df, key):
    grp = df.groupby(key)

    # last
    g = grp.last()
    g.columns = ['LAST_{}'.format(c) for c in g.columns]
    agg = g

    agg['COUNT'] = grp.size()

    a = {
        'RATIO_USED_LIMIT': ['mean', 'std', 'max', 'min'],
        'SK_DPD_PLUS': ['sum', 'mean', 'std'],
        'SHORT_PAYMENT': ['sum', 'mean', 'std'],
        'AMT_BALANCE': ['mean', 'max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'std'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum', 'mean', 'std', 'max'],
        'AMT_DRAWINGS_POS_CURRENT': ['sum', 'mean', 'std', 'max'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['sum', 'mean', 'std', 'max'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean'],
        'AMT_RECIVABLE': ['mean'],
        'AMT_TOTAL_RECEIVABLE': ['mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
        'CNT_DRAWINGS_CURRENT': ['mean'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['mean'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean'],
        'SK_DPD': ['mean'],
        'SK_DPD_DEF': ['mean'],
        'DIFF_AMT_BALANCE': ['mean'],
        'DIFF_AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        'DIFF_AMT_INST_MIN_REGULARITY': ['mean'],
        'DIFF_AMT_PAYMENT_TOTAL_CURRENT': ['mean'],
        'DIFF_AMT_RECEIVABLE_PRINCIPAL': ['mean'],
        'DIFF_AMT_TOTAL_RECEIVABLE': ['mean'],
        'DIFF_CNT_INSTALMENT_MATURE_CUM': ['mean'],
        'DIFF_SK_DPD': ['mean'],
        'DIFF_SK_DPD_DEF': ['mean'],
        'RATIO_PAYED': ['mean'],
        'FAIL_PAY_TO_PRINCIPAL': ['mean'],
        'FAIL_PAY_TO_TOTAL': ['mean'],
        'DIFF_PAYMENT_TOTAL_AND_PLAIN': ['mean'],
        'DIFF_RECEIVABLE_TOTAL_AND_PLAIN': ['mean'],
        'DIFF_RECEIVABLE_TOTAL_AND_PRINCIPAL': ['mean'],
    }
    g = grp.agg(a)
    g.columns = ['{}_{}'.format(x, y.upper()) for x, y in g.columns]
    agg = agg.join(g, on=key, how='left')

    # another features
    agg['MAX_RATIO_INSTALMENT'] =\
        agg['CNT_INSTALMENT_MATURE_CUM_MAX'] / agg['COUNT']

    # TODO: diff by max-min
    # TODO: handle NAME_CONTRACT_STATUS
    agg.columns = [
        'CRED_{}'.format(c) for c in agg.columns]

    return agg.reset_index()
