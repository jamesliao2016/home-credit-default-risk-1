import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)


def is_good(d):
    return d in ['C', '0']


def is_bad(d):
    return d not in ['C', '0', 'X']


def preprocess_bb():
    bb = pd.read_feather('./data/bureau_balance.feather')
    bb['GOOD_STATUS'] = bb['STATUS'].apply(is_good).astype('i')
    bb['BAD_STATUS'] = bb['STATUS'].apply(is_bad).astype('i')
    return bb


def main():
    bb = preprocess_bb()
    bb.to_feather('./data/bureau_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
