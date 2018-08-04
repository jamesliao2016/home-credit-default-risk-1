import pandas as pd
from utility import reduce_memory


def main():
    df = pd.read_feather('./data/pos.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    df = df.groupby('SK_ID_CURR').last()
    df = df.drop(['SK_ID_PREV'], axis=1)
    df.columns = ['POS_LAST_{}'.format(c) for c in df.columns]
    df = df.reset_index()
    reduce_memory(df)
    df.to_feather('./data/pos.last.feather')


if __name__ == '__main__':
    main()
