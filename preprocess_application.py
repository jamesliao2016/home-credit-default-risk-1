import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)


def split_org_type(df):
    def f(d):
        d = d.lower()
        d = d.replace(': ', ' ')
        d = d.split(' type ')
        m = d[0]
        t = '1' if len(d) == 1 else d[1]
        return m, t
    tmp = df['ORGANIZATION_TYPE'].apply(f)
    df['ORGANIZATION_TYPE_MAJOR'] = tmp.apply(lambda d: d[0])
    df['ORGANIZATION_TYPE_MINOR'] = tmp.apply(lambda d: d[1])


def preprocess_application():
    train_df = pd.read_feather('./data/application_train.feather')
    test_df = pd.read_feather('./data/application_test.feather')
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    split_org_type(df)

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
    df.columns = [c.replace(' ', '_') for c in df.columns]

    return df[:n_train].reset_index(), df[n_train:].reset_index()


def main():
    train_df, test_df = preprocess_application()
    train_df.to_feather('./data/application_train.preprocessed.feather')
    test_df.to_feather('./data/application_test.preprocessed.feather')


if __name__ == '__main__':
    main()
