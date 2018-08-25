import pandas as pd
from category_encoders.target_encoder import TargetEncoder


def _encode():
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    test = pd.read_feather('./data/application_test.preprocessed.feather')
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
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
    encoder.fit(df[cols], df['TARGET'])
    res = encoder.transform(df[cols])
    res.columns = ['{}_ENC'.format(c) for c in res.columns]
    res['SK_ID_CURR'] = df['SK_ID_CURR']
    res.to_feather('./data/app.enc.feather')


def main():
    _encode()


if __name__ == '__main__':
    main()
