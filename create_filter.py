import lightgbm as lgb
import pickle
from datetime import datetime
from collections import defaultdict


def _get(idx, res):
    bst = lgb.Booster(model_file='./data/lgb.model.0815-2340.{}.txt'.format(idx))
    split = bst.feature_importance('split', iteration=bst.best_iteration)
    gain = bst.feature_importance('gain', iteration=bst.best_iteration)
    feature_name = bst.feature_name()

    for f, s, g in zip(feature_name, split, gain):
        res[f][0] += s
        res[f][1] += g


def main():
    res = defaultdict(lambda: [0, 0])
    for idx in range(5):
        _get(idx, res)

    filt = []
    for k, v in res.items():
        s, g = v
        if s > 1:
            continue
        print(k, g)
        filt.append(k)

    now = datetime.now().strftime('%m%d-%H%M')
    with open('./data/filter.{}.pkl'.format(now), 'wb') as fp:
        pickle.dump(filt, fp)


if __name__ == '__main__':
    main()
