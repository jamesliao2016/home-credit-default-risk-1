import pandas as pd
from utility import calc_2nd_order_feature
pd.set_option("display.max_columns", 100)


def main():
    cre_df = pd.read_feather('./data/credit_card_balance.csv.feather')
    res_df = calc_2nd_order_feature(cre_df).set_index('SK_ID_CURR')
    res_df.columns = ['cre_{}'.format(c) for c in res_df.columns]
    res_df = res_df.reset_index()
    res_df.to_feather('./data/credit_card_balance.numeric.2nd-order.feather')


if __name__ == '__main__':
    main()
