import pandas as pd
pd.set_option("display.max_columns", 100)


def preprocess_prev(prev_df):
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


def main():
    prev_df = pd.read_feather('./data/previous_application.csv.feather')
    prev_df = prev_df.sort_values('SK_ID_CURR')
    res = preprocess_prev(prev_df)
    res = res.set_index('SK_ID_CURR')
    res.columns = [
        'prev_preprocesed_{}'.format(c) for c in res.columns]
    res = res.reset_index()
    res.to_feather('./data/preprocessed_previous_application.csv.feather')


if __name__ == '__main__':
    main()
