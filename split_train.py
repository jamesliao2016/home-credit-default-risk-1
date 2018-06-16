import numpy as np
import pandas as pd


def _split(src, dst_format):
    np.random.seed(215)
    df = pd.read_feather(src)
    pos_df = df[df['TARGET'] == 1].reset_index(drop=True)
    neg_df = df[df['TARGET'] == 0].sample(frac=1).reset_index(drop=True)
    del df
    chunck_size = len(neg_df) // 11 + 1
    for i, s in enumerate(range(0, len(neg_df), chunck_size)):
        e = min(s+chunck_size, len(neg_df))
        df = pd.concat(
            [pos_df, neg_df[s:e]]).sample(frac=1).reset_index(drop=True)
        dst = dst_format.format(i)  # noqa
        df.to_feather(dst)


def split():
    _split(
        './data/application_train.feather',
        './data/application_train.split.{}.feather',
    )


def split_preprocessed():
    _split(
        './data/application_train.preprocessed.feather',
        './data/application_train.preprocessed.split.{}.feather',
    )


def main():
    split()
    split_preprocessed()


if __name__ == '__main__':
    main()
