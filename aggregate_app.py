import pandas as pd
from utility import reduce_memory
pd.set_option("display.max_columns", 1300)
pd.set_option("display.width", 220)


def _aggregate(df, by, agg):
    print('agg {}...'.format(by))
    agg = df.groupby(by).agg(agg)
    agg.columns = ['GRP_{}_{}_{}'.format('_'.join(by), a, b.upper()) for a, b in agg.columns]
    df = df.set_index(by)
    df = df[['SK_ID_CURR']]
    df = df.join(agg, on=by, how='left')
    df = df.reset_index(drop=True)
    reduce_memory(df)
    return df.set_index('SK_ID_CURR')


def aggregate_app():
    test = pd.read_feather('./data/application_test.preprocessed.feather')
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    df = pd.concat([train, test], sort=False)
    df = df.drop(['TARGET'], axis=1)

    res = []
    fs = ['min', 'mean', 'max', 'sum', 'var']
    for by in [
        ['NAME_EDUCATION_TYPE', 'CODE_GENDER'],
        ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'],
        ['NAME_FAMILY_STATUS', 'CODE_GENDER']
    ]:
        res.append(_aggregate(df, by, {
            'AMT_CREDIT': fs,
            'AMT_ANNUITY': fs,
            'AMT_INCOME_TOTAL': fs,
            'AMT_GOODS_PRICE': fs,
            'EXT_SOURCE_1': fs,
            'EXT_SOURCE_2': fs,
            'EXT_SOURCE_3': fs,
            'OWN_CAR_AGE': fs,
            'REGION_POPULATION_RELATIVE': fs,
            'DAYS_REGISTRATION': fs,
            'CNT_CHILDREN': fs,
            'CNT_FAM_MEMBERS': fs,
            'DAYS_ID_PUBLISH': fs,
            'DAYS_BIRTH': fs,
            'DAYS_EMPLOYED': fs,
        }))

    res.append(_aggregate(df, ['CODE_GENDER', 'ORGANIZATION_TYPE'], {
        'AMT_ANNUITY': ['mean'],
        'AMT_INCOME_TOTAL': ['mean'],
        'DAYS_REGISTRATION': ['mean'],
        'EXT_SOURCE_1': ['mean'],
    }))
    res.append(_aggregate(df, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], {
        'AMT_ANNUITY': ['mean'],
        'CNT_CHILDREN': ['mean'],
        'DAYS_ID_PUBLISH': ['mean'],
    }))
    res.append(_aggregate(
        df, ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'],  {
            'EXT_SOURCE_1': ['mean'],
            'EXT_SOURCE_2': ['mean'],
        },
    ))
    res.append(_aggregate(df, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], {
        'AMT_CREDIT': ['mean'],
        'AMT_REQ_CREDIT_BUREAU_YEAR': ['mean'],
        'APARTMENTS_AVG': ['mean'],
        'BASEMENTAREA_AVG': ['mean'],
        'EXT_SOURCE_1': ['mean'],
        'EXT_SOURCE_2': ['mean'],
        'EXT_SOURCE_3': ['mean'],
        'NONLIVINGAREA_AVG': ['mean'],
        'OWN_CAR_AGE': ['mean'],
        'YEARS_BUILD_AVG': ['mean'],
    }))
    res.append(_aggregate(
        df, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], {
            'ELEVATORS_AVG': ['mean'],
            'EXT_SOURCE_1': ['mean'],
        }
    ))
    res.append(_aggregate(df, ['OCCUPATION_TYPE'], {
        'AMT_ANNUITY': ['mean'],
        'CNT_CHILDREN': ['mean'],
        'CNT_FAM_MEMBERS': ['mean'],
        'DAYS_BIRTH': ['mean'],
        'DAYS_EMPLOYED': ['mean'],
        'DAYS_ID_PUBLISH': ['mean'],
        'DAYS_REGISTRATION': ['mean'],
        'EXT_SOURCE_1': ['mean'],
        'EXT_SOURCE_2': ['mean'],
        'EXT_SOURCE_3': ['mean'],
    }))

    res = pd.concat(res, axis=1)

    return res.reset_index()


def main():
    agg = aggregate_app()
    print(agg.columns, agg.columns.shape)
    agg.to_feather('./data/app.agg.feather')


if __name__ == '__main__':
    main()
