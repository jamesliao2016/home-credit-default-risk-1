import pandas as pd
from category_encoders.target_encoder import TargetEncoder


def main():
    train_df = pd.read_feather('./data/application_train.csv.feather')
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
