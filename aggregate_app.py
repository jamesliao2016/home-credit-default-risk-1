import pandas as pd
from utility import percentile


def aggregate_app():
    test = pd.read_feather('./data/application_test.preprocessed.feather')
    train = pd.read_feather('./data/application_train.preprocessed.feather')
    agg_columns = ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'CODE_GENDER']
    df = pd.concat([train, test], sort=False)
    agg = df.groupby(agg_columns)[[
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
        'ANNUITY_LENGTH',
        'DAYS_EMPLOYED',
        'CONSUMER_GOODS_RATIO',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
    ]].agg(['mean', 'sum', 'median', 'std', 'min', 'max', percentile(20), percentile(80)])
    agg.columns = ['GRP_{}_{}_{}'.format(
        '_'.join(agg_columns), a, b.upper()) for a, b in agg.columns]
    return agg.reset_index()


def main():
    agg = aggregate_app()
    agg.to_feather('./data/application.agg.feather')


if __name__ == '__main__':
    main()
