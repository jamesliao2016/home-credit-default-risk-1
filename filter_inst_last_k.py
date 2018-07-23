import pandas as pd
from utility import filter_by_corr
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def filter_last_k(k):
    ins = pd.read_feather('./data/inst.last.{}.feather'.format(k))
    ins = filter_by_corr(ins)
    ins = ins.set_index('SK_ID_CURR')
    return ins


def main():
    res = []
    for k in [1, 5, 10, 20, 50, 100]:
        res.append(filter_last_k(k))
    res = pd.concat(res, axis=1)
    print(res.head())
    res = res.reset_index()
    res.to_feather('./data/inst.last.k.feather')


if __name__ == '__main__':
    main()
