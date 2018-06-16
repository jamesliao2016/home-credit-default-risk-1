import pandas as pd


def preprocess_bb():
    bb_df = pd.read_feather('./data/bureau_balance.feather')
    bb_df['GOOD_STATUS'] = bb_df[
        'STATUS'].apply(lambda d: d in ['C', '0']).astype('i')
    bb_df['BAD_STATUS'] = bb_df[
        'STATUS'].apply(lambda d: d not in ['C', '0', 'X']).astype('i')
    return bb_df


def main():
    bb_df = preprocess_bb()
    bb_df.to_feather('./data/bureau_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
