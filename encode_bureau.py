import pandas as pd
from utility import calc_2nd_order_feature
pd.set_option("display.max_columns", 100)


def main():
    bur_df = pd.read_feather('./data/bureau.csv.feather')
    res_df = calc_2nd_order_feature(bur_df).set_index('SK_ID_CURR')
    res_df.columns = ['bur_{}'.format(c) for c in res_df.columns]
    res_df = res_df.reset_index()
    res_df.to_feather('./data/bureau.numeric.2nd-order.feather')


if __name__ == '__main__':
    main()
