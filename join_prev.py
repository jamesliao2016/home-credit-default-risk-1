import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)


def join_prev():
    pre_df = pd.read_feather(
        './data/previous_application.preprocessed.feather')
    cre_df = pd.read_feather('./data/credit_card_balance.agg.feather')
    pos_df = pd.read_feather('./data/POS_CASH_balance.agg.feather')
    key = ['SK_ID_CURR', 'SK_ID_PREV']
    pre_df = pre_df.merge(cre_df, on=key, how='outer')
    pre_df = pre_df.merge(pos_df, on=key, how='outer')

    return pre_df


def main():
    pre_df = join_prev()
    pre_df.to_feather('./data/previous_application.joined.feather')


if __name__ == '__main__':
    main()
