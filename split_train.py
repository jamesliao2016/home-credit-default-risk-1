import numpy as np
import pandas as pd


def split_preprocessed():
    df = pd.read_feather('./data/application_train.preprocessed.feather')
    pos_df = df[df['TARGET'] == 1].reset_index(drop=True)
    neg_df = df[df['TARGET'] == 0].sample(frac=1).reset_index(drop=True)
    del df
    chunck_size = len(neg_df) // 11 + 1
    for i, s in enumerate(range(0, len(neg_df), chunck_size)):
        e = min(s+chunck_size, len(neg_df))
        df = pd.concat(
            [pos_df, neg_df[s:e]]).sample(frac=1).reset_index(drop=True)
        dst = './data/application_train.preprocessed.split.{}.feather'.format(i)  # noqa
        df.to_feather(dst)


def split():
    df = pd.read_feather('./data/application_train.feather')
    pos_df = df[df['TARGET'] == 1].reset_index(drop=True)
    neg_df = df[df['TARGET'] == 0].sample(frac=1).reset_index(drop=True)
    del df
    chunck_size = len(neg_df) // 11 + 1
    for i, s in enumerate(range(0, len(neg_df), chunck_size)):
        e = min(s+chunck_size, len(neg_df))
        df = pd.concat(
            [pos_df, neg_df[s:e]]).sample(frac=1).reset_index(drop=True)
        dst = './data/application_train.split.{}.feather'.format(i)
        df.to_feather(dst)


def main():
    np.random.seed(215)
    split()
    np.random.seed(215)
    split_preprocessed()


if __name__ == '__main__':
    main()
