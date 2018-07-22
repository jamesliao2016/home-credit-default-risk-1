import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)


def preprocess_prev():
    pre_df = pd.read_feather(
        './data/previous_application.preprocessed.feather')

    # TODO: Move preprocess prev
    pre_df['IS_REFUSED'] =\
        (pre_df['NAME_CONTRACT_STATUS'] == 'Refused').astype('i')

    pre_df = pre_df.sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
    pre_df, cat_columns = one_hot_encoder(pre_df)

    grp = pre_df.groupby('SK_ID_CURR')
    g = grp[['SK_ID_PREV']].count()
    g.columns = ['COUNT']
    pre_agg = g

    # REFUSED RATIO
    g = grp[['IS_REFUSED']].sum()
    pre_agg = pre_agg.join(g, on='SK_ID_CURR', how='left')
    pre_agg['RATIO_REFUSED'] = pre_agg['IS_REFUSED'] / pre_agg['COUNT']

    # NOT COMPLETE RATIO
    g = pre_df[
        pre_df['DAYS_LAST_DUE_1ST_VERSION'] < 365243.0].reset_index()
    g['NOT_COMPLETE'] = (g['DAYS_LAST_DUE_1ST_VERSION'] >= 0).astype('i')
    g = g.groupby('SK_ID_CURR')
    g = g[['NOT_COMPLETE']].sum()
    pre_agg = pre_agg.join(g, on='SK_ID_CURR', how='left')
    pre_agg['RATIO_NOT_COMPLETE'] = pre_agg['NOT_COMPLETE'] / pre_agg['COUNT']

    # Previous applications numeric features
    fs = ['sum', 'median', 'mean', 'std', 'max', 'min']
    agg = {
        'AMT_ANNUITY': fs,
        'AMT_APPLICATION': fs,
        'AMT_CREDIT': fs,
        'RATIO_APP_TO_CREDIT': fs,
        'AMT_DOWN_PAYMENT': fs,
        'AMT_GOODS_PRICE': fs,
        'HOUR_APPR_PROCESS_START': fs,
        'RATE_DOWN_PAYMENT': fs,
        'DAYS_DECISION': fs,
        'CNT_PAYMENT': fs,
    }
    for c in cat_columns:
        agg[c] = ['mean']

    for c in pre_df.columns:
        if c == 'SK_ID_CURR' or c == 'SK_ID_PREV':
            continue
        if c in agg:
            continue
        agg[c] = ['mean']

    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    pre_agg = pre_agg.join(g, on='SK_ID_CURR', how='left')

    # Previous Applications: Approved Applications
    approved = pre_df[pre_df['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(agg)
    approved_agg.columns = [
        'APPROVED_{}_{}'.format(
                a.upper(), b) for a, b in approved_agg.columns]

    pre_agg = pre_agg.join(approved_agg, on='SK_ID_CURR', how='left')

    # Previous Applications: Refused Applications
    refused = pre_df[pre_df['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(agg)
    refused_agg.columns = [
        'REFUSED_{}_{}'.format(a.upper(), b) for a, b in refused_agg.columns]
    pre_agg = pre_agg.join(refused_agg, on='SK_ID_CURR', how='left')

    pre_agg.columns = ['PREV_{}'.format(c) for c in pre_agg.columns]

    return pre_agg.reset_index()


def main():
    agg = preprocess_prev()
    reduce_memory(agg)
    agg.to_feather('./data/previous_application.agg.feather')


if __name__ == '__main__':
    main()
