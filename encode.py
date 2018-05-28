import pandas as pd
from category_encoders.target_encoder import TargetEncoder


def encode_train(train_df, test_df):
    cols = [
        'CODE_GENDER',
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',  # Level of highest education the client achieved,  # noqa
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'FLAG_MOBIL',
        'FLAG_EMP_PHONE',
        'FLAG_WORK_PHONE',
        'FLAG_CONT_MOBILE',
        'FLAG_PHONE',
        'FLAG_EMAIL',
        'OCCUPATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START',
        'HOUR_APPR_PROCESS_START',
        'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY',
        'ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE',
        'NAME_CONTRACT_TYPE',  # Identification if loan is cash or revolving,
    ]
    encoder = TargetEncoder(cols=cols)
    encoder.fit(train_df[cols], train_df['TARGET'])
    encoded_train_df = encoder.transform(train_df[cols])
    encoded_test_df = encoder.transform(test_df[cols])
    encoded_train_df.columns = ['ENCODED_{}'.format(c) for c in cols]
    encoded_test_df.columns = ['ENCODED_{}'.format(c) for c in cols]
    pd.concat([train_df, encoded_train_df], axis=1).to_feather(
        './data/application_train.csv.encoded.feather')
    pd.concat([test_df, encoded_test_df], axis=1).to_feather(
        './data/application_test.csv.encoded.feather')


def main():
    train_df = pd.read_feather('./data/application_train.csv.feather')
    test_df = pd.read_feather('./data/application_test.csv.feather')
    encode_train(train_df, test_df)

    train_df = train_df[['SK_ID_CURR', 'TARGET']]
    prev_df = pd.read_feather('./data/previous_application.csv.feather')
    df = train_df.merge(prev_df, on='SK_ID_CURR')

    col = 'SELLERPLACE_AREA'
    encoder = TargetEncoder(cols=[col])
    encoder.fit(df[[col]], df['TARGET'])
    prev_df['ENCODED_{}'.format(col)] = encoder.transform(
        prev_df[[col]])[col]

    prev_df.to_feather('./data/previous_application.csv.encoded.feather')


if __name__ == '__main__':
    main()
