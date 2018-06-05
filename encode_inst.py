import pandas as pd
from utility import calc_2nd_order_feature
pd.set_option("display.max_columns", 100)


def main():
    inst_df = pd.read_feather('./data/installments_payments.csv.feather')
    res_df = calc_2nd_order_feature(inst_df).set_index('SK_ID_CURR')
    res_df.columns = ['inst_{}'.format(c) for c in res_df.columns]
    res_df = res_df.reset_index()
    res_df.to_feather('./data/installments_payments.numeric.2nd-order.feather')


if __name__ == '__main__':
    main()
