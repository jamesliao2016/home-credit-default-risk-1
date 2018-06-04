import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def handle_numeric(df, res, target):
    grp = df.groupby('SK_ID_CURR')
    grp = grp[[target]].agg(
        ['sum', 'mean', 'std', 'min', 'max'])
    grp.columns = ['{}_{}'.format(
        target, c) for c in ['sum', 'mean', 'std', 'min', 'max']]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
    return res


def preprocess_credit(cred_df):
    grp = cred_df.groupby('SK_ID_CURR')
    grp = grp[['SK_ID_PREV']].count()
    grp.columns = ['CREDIT_COUNT']
    res = grp.reset_index()

    grp = cred_df.groupby(
        ['SK_ID_CURR', 'SK_ID_PREV'])[['CNT_INSTALMENT_MATURE_CUM']].max()
    grp.columns = ['INSTALMENT_COUNT']
    grp = grp.reset_index().groupby('SK_ID_CURR').sum().reset_index()
    res = res.merge(grp, on='SK_ID_CURR', how='left')
    res['INSTALMENT_ratio'] = res['INSTALMENT_COUNT'] / res['CREDIT_COUNT']

    grp = cred_df.groupby(
        ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']
    )[['AMT_BALANCE']].max().reset_index('AMT_CREDIT_LIMIT_ACTUAL')
    grp['USED_LIMIT'] = grp['AMT_BALANCE'] / grp['AMT_CREDIT_LIMIT_ACTUAL']
    grp = grp[['USED_LIMIT']].reset_index()
    res = handle_numeric(grp, res, 'USED_LIMIT')

    cred_df['SK_DPD_PLUS'] = (cred_df['SK_DPD'] > 0).astype('i')
    grp = cred_df.groupby(
        ['SK_ID_CURR', 'SK_ID_PREV'])[['SK_DPD_PLUS']].sum()
    grp = grp.reset_index()
    res = handle_numeric(grp, res, 'SK_DPD_PLUS')

    cred_df['SHORT_PAYMENT'] = (
        cred_df['AMT_INST_MIN_REGULARITY'] > cred_df['AMT_PAYMENT_CURRENT']
    ).astype('i')
    grp = cred_df.groupby('SK_ID_CURR')['SHORT_PAYMENT'].sum()
    grp = grp.reset_index()
    res = res.merge(grp, on='SK_ID_CURR', how='left')
    res['SHORT_PAYMENT_ratio'] = res['SHORT_PAYMENT'] / res['CREDIT_COUNT']

    grp = cred_df.groupby('SK_ID_CURR')[[
        'AMT_DRAWINGS_CURRENT',
        'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT',
        'AMT_DRAWINGS_OTHER_CURRENT',
    ]].sum()
    grp['AMT_DRAWINGS_ATM_CURRENT'] /= grp['AMT_DRAWINGS_CURRENT']
    grp['AMT_DRAWINGS_POS_CURRENT'] /= grp['AMT_DRAWINGS_CURRENT']
    grp['AMT_DRAWINGS_OTHER_CURRENT'] /= grp['AMT_DRAWINGS_CURRENT']
    grp.pop('AMT_DRAWINGS_CURRENT')
    grp.columns = ['{}_ratio'.format(c) for c in grp.columns]
    grp = grp.reset_index()
    res = res.merge(grp, on='SK_ID_CURR', how='left')

    grp = cred_df.groupby('SK_ID_CURR')[[
        'AMT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_CURRENT',
    ]].sum()
    grp['AMT_DRAWINGS_CURRENT'] /= grp['CNT_DRAWINGS_CURRENT']
    grp.pop('CNT_DRAWINGS_CURRENT')
    grp.columns = ['MEAN_AMT_DRAWINGS_CURRENT']
    grp = grp.reset_index()
    res = res.merge(grp, on='SK_ID_CURR', how='left')

    return res


def main():
    cred_df = pd.read_feather('./data/credit_card_balance.csv.feather')
    cred_df = cred_df.sort_values('SK_ID_CURR').reset_index()
    res = preprocess_credit(cred_df)
    res = res.set_index('SK_ID_CURR')
    res.columns = [
        'credit_preprocesed_{}'.format(c) for c in res.columns]
    res = res.reset_index()
    res.to_feather('./data/preprocessed_credit_balance.csv.feather')


if __name__ == '__main__':
    main()
