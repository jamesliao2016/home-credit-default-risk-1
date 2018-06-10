import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)


def concat(s):
    return ' '.join(s.values.tolist())


def aggregate(df, base):
    numeric_columns = df.select_dtypes(np.number).columns.tolist()
    for c in ['SK_ID_CURR', 'SK_ID_PREV']:
        while c in numeric_columns:
            numeric_columns.remove(c)
    categorical_columns = df.select_dtypes('object').columns.tolist()
    for c in categorical_columns:
        df[c] = df[c].astype(
            'category').cat.codes.apply(lambda d: 'c{}'.format(d))

    numeric_agg = {}
    for c in numeric_columns:
        numeric_agg[c] = ['min', 'max', 'mean', 'sum', 'var']

    grp = df.groupby(base)
    cat_agg = grp[categorical_columns].agg(concat)

    num_agg = grp[numeric_columns].agg(numeric_agg)
    num_agg.columns = [
        a + "_" + b.upper() for a, b in num_agg.columns.tolist()]

    if len(categorical_columns) > 0:
        agg = cat_agg.join(num_agg, on=base)
    else:
        agg = num_agg
    return agg


def concat_bureau():
    bb_df = pd.read_feather('./data/bureau_balance.csv.feather')
    bb_agg = aggregate(bb_df, ['SK_ID_BUREAU'])
    bb_agg.columns = ['BB_{}'.format(c) for c in bb_agg.columns]
    bure_df = pd.read_feather('./data/bureau.csv.feather')
    bure_df = bure_df.set_index('SK_ID_BUREAU')
    bure_df = bure_df.join(bb_agg, on='SK_ID_BUREAU', how='left')
    bb_status = bure_df[['SK_ID_CURR', 'BB_STATUS']].reset_index(drop=True)
    del bure_df['BB_STATUS']
    bb_status['BB_STATUS'].fillna("", inplace=True)
    bb_status['SK_ID_CURR'] = bb_status['SK_ID_CURR'].astype('i')
    bb_status = bb_status.groupby('SK_ID_CURR')[['BB_STATUS']].agg(concat)

    bure_df['SK_ID_CURR'] = bure_df['SK_ID_CURR'].astype('i')
    bure_agg = aggregate(bure_df, ['SK_ID_CURR'])
    bure_agg.columns = ['BURE_{}'.format(c) for c in bure_agg.columns]
    bure_agg = bure_agg.join(bb_status, on='SK_ID_CURR', how='left')
    bure_agg.reset_index().to_feather('./data/bureau.concat.feather')


def concat_df(in_path, prefix):
    df = pd.read_feather(in_path)
    agg = aggregate(df, ['SK_ID_CURR'])
    agg.columns = ['{}_{}'.format(prefix, c) for c in agg.columns]
    agg = agg.reset_index()
    out_path = in_path.replace('csv', 'concat')
    print(agg.head())
    agg.to_feather(out_path)


def concat_app():
    def handle_category(df):
        categorical_columns = df.select_dtypes('object').columns.tolist()
        for c in categorical_columns:
            df[c] = df[c].astype(
                'category').cat.codes.apply(lambda d: 'c{}'.format(d))
        return df

    train_df = pd.read_feather('./data/application_train.csv.feather')
    test_df = pd.read_feather('./data/application_test.csv.feather')
    df = train_df.append(test_df).reset_index(drop=True)
    df = handle_category(df)
    for in_path in [
        './data/bureau.concat.feather',
        './data/credit_card_balance.concat.feather',
        './data/POS_CASH_balance.concat.feather',
        './data/installments_payments.concat.feather',
        './data/previous_application.concat.feather',
    ]:
        print('load {}...'.format(in_path))
        other_df = pd.read_feather(in_path, nthreads=12)
        df = df.merge(other_df, on='SK_ID_CURR', how='left')
    train_df, test_df = df[
        :len(train_df)].reset_index(), df[len(train_df):].reset_index()
    train_df['TARGET'] = train_df['TARGET'].astype('i')
    train_df.to_feather('./data/application_train.concat.feather')
    test_df.to_feather('./data/application_test.concat.feather')


def main():
    concat_bureau()
    concat_df('./data/credit_card_balance.csv.feather', 'CREDIT')
    concat_df('./data/POS_CASH_balance.csv.feather', 'POS')
    concat_df('./data/installments_payments.csv.feather', 'INS')
    concat_df('./data/previous_application.csv.feather', 'PREV')
    concat_app()


if __name__ == '__main__':
    main()
