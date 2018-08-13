import lightgbm as lgb
import pickle
from datetime import datetime
from collections import defaultdict


def _get(idx, res):
    bst = lgb.Booster(model_file='./data/lgb.model.{}.txt'.format(idx))
    split = bst.feature_importance('split', iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for f, s in zip(feature_name, split):
        res[f] += s


def main():
    res = defaultdict(int)
    for idx in range(5):
        _get(idx, res)

    filt = []
    for k, v in res.items():
        if v > 1:
            continue
        filt.append(k)

    now = datetime.now().strftime('%m%d-%H%M')
    with open('./data/filter.{}.pkl'.format(now), 'wb') as fp:
        pickle.dump(filt, fp)


if __name__ == '__main__':
    main()
