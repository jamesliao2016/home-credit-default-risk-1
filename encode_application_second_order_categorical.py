import gc
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
pd.set_option("display.max_columns", 100)


def main():
    train_df = pd.read_feather('./data/application_train.csv.feather')
    test_df = pd.read_feather('./data/application_test.csv.feather')

    columns = train_df.select_dtypes('object').columns.tolist()
    train_df = train_df[columns + ['TARGET']]
    train_df.fillna('__NaN__', inplace=True)
    test_df = test_df[columns]
    test_df.fillna('__NaN__', inplace=True)
    train_id = train_df.pop('SK_ID_CURR')
    test_id = test_df.pop('SK_ID_CURR')
    target = train_df['TARGET']
    train_res = train_df[[]].copy()
    test_res = test_df[[]].copy()
    for i in range(len(columns)):
        a = columns[i]
        for j in range(i+1, len(columns)):
            b = columns[j]
            c = 'ENCODED_{}__{}'.format(a, b)
            train_res[c] = train_df[a] + '--' + train_df[b]
            test_res[c] = test_df[a] + '--' + test_df[b]
            gc.collect()

    encoder = TargetEncoder()
    train_res = encoder.fit_transform(train_res, target)
    train_res['SK_ID_CURR'] = train_id
    test_res = encoder.transform(test_res)
    test_res['SK_ID_CURR'] = test_id
    train_res.to_feather(
        './data/application_train.2nd-order-categorical.feather')
    test_res.to_feather(
        './data/application_test.2nd-order-categorical.feather')


if __name__ == '__main__':
    main()
