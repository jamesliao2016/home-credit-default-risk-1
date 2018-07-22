import pandas as pd
from utility import one_hot_encoder
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def main():
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    columns = [c for c in df.columns if df[c].dtype == 'object']
    df = df[['SK_ID_CURR']+columns]
    df, _ = one_hot_encoder(df)
    df = df.groupby('SK_ID_CURR').sum()
    df.columns = ['BURE_{}'.format(c) for c in df.columns]
    df = df.reset_index()
    df.to_feather('./data/bureau.agg.cat.feather')


if __name__ == '__main__':
    main()
