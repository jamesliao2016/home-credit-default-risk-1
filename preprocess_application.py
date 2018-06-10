import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 100)


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(
        df, columns=categorical_columns, dummy_na=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def preprocess_application(train_df, test_df):
    df = train_df.append(test_df).reset_index(drop=True)
    # from jsaguiar/lightgbm-with-simple-features-0-785-lb
    for f in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[f], uniques = pd.factorize(df[f])
    df, cat_cols = one_hot_encoder(df)
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df[:len(train_df)].reset_index(), df[len(train_df):].reset_index()


def main():
    train_df = pd.read_feather('./data/application_train.csv.feather')
    test_df = pd.read_feather('./data/application_test.csv.feather')
    train_df, test_df = preprocess_application(train_df, test_df)
    train_df.to_feather('./data/application_train.preprocessed.feather')
    test_df.to_feather('./data/application_test.preprocessed.feather')


if __name__ == '__main__':
    main()
