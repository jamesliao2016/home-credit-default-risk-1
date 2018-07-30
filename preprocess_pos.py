import pandas as pd
from utility import reduce_memory
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 200)


def preprocess_pos():
    df = pd.read_feather('./data/POS_CASH_balance.feather')

    # add features
    df['RATIO_CNT_INST'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT']

    return df


def main():
    df = preprocess_pos()
    reduce_memory(df)
    df.to_feather('./data/pos.preprocessed.feather')


if __name__ == '__main__':
    main()
