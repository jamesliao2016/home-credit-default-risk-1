import pandas as pd


def is_good(d):
    return d in ['C', '0']


def is_bad(d):
    return d not in ['C', '0', 'X']


def preprocess_bb():
    bb_df = pd.read_feather('./data/bureau_balance.feather')
    bb_df['GOOD_STATUS'] = bb_df['STATUS'].apply(is_good).astype('i')
    bb_df['BAD_STATUS'] = bb_df['STATUS'].apply(is_bad).astype('i')
    return bb_df


def main():
    bb_df = preprocess_bb()
    bb_df.to_feather('./data/bureau_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
