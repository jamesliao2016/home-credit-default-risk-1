import pandas as pd
pd.set_option("display.max_columns", 100)


def handle_category(bure_df, res, target):
    grp = bure_df.groupby('SK_ID_CURR')
    grp = grp[[target]].nunique()
    grp.columns = ['{}_nunique'.format(target)]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
    res['{}_diversity'.format(target)] = res[
        '{}_nunique'.format(target)] / res['BUREAU_COUNT']

    categories = bure_df[target].unique().tolist()
    for c in categories:
        grp = bure_df[bure_df[target] == c].groupby('SK_ID_CURR')
        grp = grp[[target]].count()
        col = '{}_{}_ratio'.format(target, c).replace(' ', '_')
        grp.columns = [col]
        res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
        res[col].fillna(0, inplace=True)
        res[col] /= res['BUREAU_COUNT']

    return res


def handle_numeric(bure_df, res, target):
    grp = bure_df.groupby('SK_ID_CURR')
    grp = grp[[target]].agg(
        ['sum', 'mean', 'std', 'min', 'max'])
    grp.columns = ['{}_{}'.format(
        target, c) for c in ['sum', 'mean', 'std', 'min', 'max']]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
    return res


def preprocess_bureau(bure_df):
    grp = bure_df.groupby('SK_ID_CURR')
    grp = grp[['SK_ID_BUREAU']].count()
    grp.columns = ['BUREAU_COUNT']
    res = grp.reset_index()

    tmp = bure_df[['SK_ID_CURR', 'DAYS_CREDIT']]
    tmp = tmp.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], ascending=False)
    grp = tmp.groupby('SK_ID_CURR').shift(1)
    tmp['DAYS_CREDIT_diff'] = tmp['DAYS_CREDIT'] - grp['DAYS_CREDIT']
    null_idx = tmp[pd.isnull(tmp['DAYS_CREDIT_diff'])].index
    tmp.loc[null_idx, 'DAYS_CREDIT_diff'] = tmp.loc[null_idx, 'DAYS_CREDIT']
    grp = tmp.groupby('SK_ID_CURR')[['DAYS_CREDIT_diff']].agg(
        ['mean', 'std', 'min', 'max'])
    grp.columns = ['DAYS_CREDIT_DIFF_{}'.format(
        c) for c in ['mean', 'std', 'min', 'max']]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')

    bure_df['DAYS_CREDIT_ENDDATE_PLUS'] = (
        bure_df['DAYS_CREDIT_ENDDATE'] >= 0).astype('i')
    grp = bure_df.groupby('SK_ID_CURR')
    grp = grp[['DAYS_CREDIT_ENDDATE_PLUS']].sum()
    grp.columns = ['DAYS_CREDIT_ENDDATE_PLUS_count']
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')
    res['DAYS_CREDIT_ENDDATE_PLUS_ratio'] = res[
        'DAYS_CREDIT_ENDDATE_PLUS_count'] / res['BUREAU_COUNT']

    tmp = bure_df[bure_df['DAYS_CREDIT_ENDDATE_PLUS'] > 0]
    tmp = tmp[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE']]
    tmp = tmp.sort_values(['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE'])
    grp = tmp.groupby('SK_ID_CURR').shift(1)
    tmp['DAYS_CREDIT_ENDDATE_diff'] = tmp[
        'DAYS_CREDIT_ENDDATE'] - grp['DAYS_CREDIT_ENDDATE']
    null_idx = tmp[pd.isnull(tmp['DAYS_CREDIT_ENDDATE_diff'])].index
    tmp.loc[null_idx, 'DAYS_CREDIT_ENDDATE_diff'] = tmp.loc[
        null_idx, 'DAYS_CREDIT_ENDDATE']
    grp = tmp.groupby('SK_ID_CURR')[['DAYS_CREDIT_ENDDATE_diff']].agg(
        ['mean', 'std', 'min', 'max'])
    grp.columns = ['DAYS_CREDIT_ENDDATE_DIFF_{}'.format(
        c) for c in ['mean', 'std', 'min', 'max']]
    res = res.merge(grp.reset_index(), on='SK_ID_CURR', how='left')

    bure_df['AMT_CREDIT_SUM'].fillna(0, inplace=True)
    bure_df['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
    bure_df['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
    bure_df['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)
    res = handle_numeric(bure_df, res, 'AMT_CREDIT_SUM')
    res = handle_numeric(bure_df, res, 'AMT_CREDIT_SUM_DEBT')
    res = handle_numeric(bure_df, res, 'AMT_CREDIT_SUM_OVERDUE')
    res = handle_numeric(bure_df, res, 'CNT_CREDIT_PROLONG')
    res['CREDIT_DEBT_ratio'] = res[
        'AMT_CREDIT_SUM_DEBT_sum'] / res['AMT_CREDIT_SUM_sum']
    res['CREDIT_OVERDUE_ratio'] = res[
        'AMT_CREDIT_SUM_OVERDUE_sum'] / res['AMT_CREDIT_SUM_DEBT_sum']

    res = handle_category(bure_df, res, 'CREDIT_CURRENCY')
    res = handle_category(bure_df, res, 'CREDIT_ACTIVE')
    res = handle_category(bure_df, res, 'CREDIT_TYPE')

    return res


def main():
    bure_df = pd.read_feather('./data/bureau.csv.feather')
    res = preprocess_bureau(bure_df)
    res = res.set_index('SK_ID_CURR')
    res.columns = [
        'bureau_preprocesed_{}'.format(c) for c in res.columns]
    res = res.reset_index()
    res.to_feather('./data/preprocessed_bureau.csv.feather')


if __name__ == '__main__':
    main()
