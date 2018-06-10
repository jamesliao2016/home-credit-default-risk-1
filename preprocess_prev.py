import numpy as np
import pandas as pd
from utility import one_hot_encoder
pd.set_option("display.max_columns", 100)


def create_new_features(prev_df):
    grp = prev_df.groupby('SK_ID_CURR')
    grp = grp[['SK_ID_PREV']].count()
    grp.columns = ['PREV_COUNT']
    res = grp.reset_index()

    # REFUSED RATIO
    grp = prev_df
    grp['IS_REFUSED'] = (grp['NAME_CONTRACT_STATUS'] == 'Refused').astype('i')
    grp = grp.groupby('SK_ID_CURR')
    grp = grp[['IS_REFUSED']].sum()
    res = res.merge(grp, on='SK_ID_CURR', how='left')
    res['REFUSED_ratio'] = res['IS_REFUSED'] / res['PREV_COUNT']

    # NOT COMPLETE RATIO
    grp = prev_df[
        prev_df['DAYS_LAST_DUE_1ST_VERSION'] < 365243.0].reset_index()
    grp['NOT_COMPLETE'] = (grp['DAYS_LAST_DUE_1ST_VERSION'] >= 0).astype('i')
    grp = grp.groupby('SK_ID_CURR')
    grp = grp[['NOT_COMPLETE']].sum()
    res = res.merge(grp, on='SK_ID_CURR', how='left')
    res['NOT_COMPLETE_ratio'] = res['NOT_COMPLETE'] / res['PREV_COUNT']

    return res


def preprocess_prev(prev_df):
    prev_df, cat_cols = one_hot_encoder(prev_df)
    # Days 365.243 values -> nan
    prev_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev_df[
        'APP_CREDIT_PERC'] = prev_df['AMT_APPLICATION'] / prev_df['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev_df.groupby(
        'SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = [
        'PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns]
    # Previous Applications: Approved Applications - only numerical features
    approved = prev_df[prev_df['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = [
        'APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns]
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev_df[prev_df['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = [
        'REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns]
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    return prev_agg


def main():
    prev_df = pd.read_feather('./data/previous_application.csv.feather')
    prev_df = prev_df.sort_values('SK_ID_CURR')
    prev_agg = preprocess_prev(prev_df)
    res = create_new_features(prev_df)
    res = res.merge(prev_agg, how='left', on='SK_ID_CURR')
    res = res.set_index('SK_ID_CURR')
    res.columns = [
        'PREV_{}'.format(c.replace(' ', '_')) for c in res.columns]
    res = res.reset_index()
    res.to_feather('./data/previous_application.preprocessed.feather')


if __name__ == '__main__':
    main()
