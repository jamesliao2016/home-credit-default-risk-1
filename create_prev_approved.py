import pandas as pd
from utility import one_hot_encoder, reduce_memory
pd.set_option("display.max_columns", 100)


def _create():
    df = pd.read_feather('./data/prev.preprocessed.feather')
    df, cat_columns = one_hot_encoder(df)
    df.columns = [c.replace(' ', '_') for c in df.columns]

    fs = ['sum', 'median', 'mean', 'std', 'max', 'min']
    a = {
        'SK_ID_PREV': ['nunique'],
        'AMT_ANNUITY': fs,
        'AMT_APPLICATION': fs,
        'AMT_CREDIT': fs,
        'RATIO_APP_TO_CREDIT': fs,
        'AMT_DOWN_PAYMENT': fs,
        'AMT_GOODS_PRICE': fs,
        'HOUR_APPR_PROCESS_START': fs,
        'RATE_DOWN_PAYMENT': fs,
        'DAYS_DECISION': fs,
        'CNT_PAYMENT': fs,
        'NAME_CLIENT_TYPE_New': ['mean'],
        'NAME_YIELD_GROUP_middle': ['mean'],
        'NAME_YIELD_GROUP_low_action': ['mean'],
        'NAME_TYPE_SUITE_Unaccompanied': ['mean'],
        'NAME_TYPE_SUITE_nan': ['mean'],
        'PRODUCT_COMBINATION_Cash_X-Sell:_low': ['mean'],
        'PRODUCT_COMBINATION_POS_industry_with_interest': ['mean'],
        'PRODUCT_COMBINATION_POS_household_with_interest': ['mean'],
        'PRODUCT_COMBINATION_POS_mobile_with_interest': ['mean'],
        'NAME_SELLER_INDUSTRY_Connectivity': ['mean'],
        'NAME_SELLER_INDUSTRY_Consumer_electronics': ['mean'],
        'NAME_SELLER_INDUSTRY_Furniture': ['mean'],
        # added
        'NOT_COMPLETE': ['mean'],
    }

    df = df[df['FLAG_Approved'] == 1]
    g = df.groupby('SK_ID_CURR').agg(a)
    g.columns = ['PREV_APPROVED_{}_{}'.format(a, b.upper()) for a, b in g.columns]

    return g.reset_index()


def main():
    agg = _create()
    reduce_memory(agg)
    agg.to_feather('./data/prev.approved.feather')


if __name__ == '__main__':
    main()
