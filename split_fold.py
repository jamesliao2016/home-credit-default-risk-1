import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    train = pd.read_feather('./data/application_train.feather')
    cv = StratifiedKFold(5, True, 215)
    for i, (train_idx, valid_idx) in enumerate(cv.split(train, train['TARGET'])):
        valid = train.loc[valid_idx, ['SK_ID_CURR']].reset_index()
        valid.to_feather('./data/fold.{}.feather'.format(i))


if __name__ == '__main__':
    main()
