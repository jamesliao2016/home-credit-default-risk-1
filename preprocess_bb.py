import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)


def is_good(d):
    return d in ['C', '0']


def is_bad(d):
    return d not in ['C', '0', 'X']


def status_to_dpd(d):
    if d in ['C', 'X']:
        return 0
    return int(d)


def convert_status(d):
    if d in ['C', 'X']:
        return d
    return 'O'


def preprocess_bb():
    bb = pd.read_feather('./data/bureau_balance.feather')
    bb['DPD'] = bb['STATUS'].apply(status_to_dpd)
    bb['STATUS'] = bb['STATUS'].apply(convert_status)
    return bb


def main():
    bb = preprocess_bb()
    bb.to_feather('./data/bureau_balance.preprocessed.feather')


if __name__ == '__main__':
    main()
