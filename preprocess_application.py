import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)


def preprocess_application():
    train_df = pd.read_feather('./data/application_train.feather')
    test_df = pd.read_feather('./data/application_test.feather')
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['LOAN_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_LENGTH'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['WORKING_LIFE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_FAM'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # from https://www.kaggle.com/shep312/lightgbm-harder-better-slower
    df['CONSUMER_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['ANN_LENGTH_EMPLOYED_RATIO'] =\
        df['ANNUITY_LENGTH'] / df['DAYS_EMPLOYED']
    df['TOTAL_DOCS_SUBMITTED'] =\
        df.loc[:, df.columns.str.contains('FLAG_DOCUMENT')].sum(axis=1)

    # from ogrellier/fork-lightgbm-with-simple-features
    inc_by_org = df[
        ['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
            'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if (
        'FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    df['DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['ANNUITY_TO_INCOME_RATIO'] =\
        df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['SOURCES_PROD'] =\
        df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] =\
        df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['SCORES_STD'] =\
        df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['SCORES_STD'] = df['SCORES_STD'].fillna(df['SCORES_STD'].mean())
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['PHONE_TO_BIRTH_RATIO'] =\
        df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['PHONE_TO_EMPLOY_RATIO'] =\
        df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']

    df.columns = [c.replace(' ', '_') for c in df.columns]

    train_df = df[:n_train].reset_index(drop=True)
    test_df = df[n_train:].reset_index(drop=True)

    return train_df,  test_df


def main():
    train_df, test_df = preprocess_application()
    train_df.to_feather('./data/application_train.preprocessed.feather')
    test_df.to_feather('./data/application_test.preprocessed.feather')


if __name__ == '__main__':
    main()
