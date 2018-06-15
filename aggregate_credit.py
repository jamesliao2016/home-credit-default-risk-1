import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)
# TODO: Add last status


def handle_numeric(df, res, target):
    grp = df.groupby('SK_ID_CURR')
    grp = grp[[target]].agg(
        ['sum', 'mean', 'std', 'min', 'max'])
    grp.columns = ['{}_{}'.format(
        target, c) for c in ['sum', 'mean', 'std', 'min', 'max']]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
    return res


def aggregate_credit():
    key = ['SK_ID_CURR', 'SK_ID_PREV']
    cred_df = pd.read_feather(
        './data/credit_card_balance.preprocessed.feather')

    # TODO: move to preprocessing
    cred_df['SK_DPD_PLUS'] = (cred_df['SK_DPD'] > 0).astype('i')
    cred_df['SHORT_PAYMENT'] = (
        cred_df['AMT_INST_MIN_REGULARITY'] >
        cred_df['AMT_PAYMENT_TOTAL_CURRENT']
    ).astype('i')

    # aggregate
    grp = cred_df.groupby(key)
    g = grp[['MONTHS_BALANCE']].count()
    g.columns = ['COUNT']
    cred_agg = g.reset_index()

    g = grp[['CNT_INSTALMENT_MATURE_CUM']].max()
    g.columns = ['MAX_CNT_INSTALMENT']
    cred_agg = cred_agg.merge(g, on=key, how='left')
    cred_agg['MAX_RATIO_INSTALMENT'] =\
        cred_agg['MAX_CNT_INSTALMENT'] / cred_agg['COUNT']

    agg = {
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
        'CNT_INSTALMENT_MATURE_CUM': ['mean'],
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
    g = grp.agg(agg)
    g.columns = ['{}_{}'.format(a.upper(), b) for a, b in g.columns]
    cred_agg = cred_agg.merge(g, on=key, how='left')
    # TODO: diff by max-min
    # TODO: handle NAME_CONTRACT_STATUS

    return cred_agg


def main():
    cred_agg = aggregate_credit()
    cred_agg = cred_agg.set_index('SK_ID_CURR')
    cred_agg.columns = [
        'CRED_{}'.format(c) for c in cred_agg.columns]
    cred_agg = cred_agg.reset_index()
    cred_agg.to_feather('./data/credit_balance.agg.feather')


if __name__ == '__main__':
    main()
