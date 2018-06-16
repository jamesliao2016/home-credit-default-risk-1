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
