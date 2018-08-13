import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def has_x_sell(d):
    if pd.isnull(d):
        return 0
    if d.lower().find('x-sell') >= 0:
        return 1
    return 0


def preprocess_prev():
    df = pd.read_feather('./data/previous_application.feather')

    df.pop('NAME_GOODS_CATEGORY')

    df['FLAG_PURPOSE_NA'] = (
        (df['NAME_CASH_LOAN_PURPOSE'] == 'XNA') | (df['NAME_CASH_LOAN_PURPOSE'] == 'XAP')
    ).astype('i')
    df.pop('NAME_CASH_LOAN_PURPOSE')

    df['FLAG_THROUGH_BANK'] = (df['NAME_PAYMENT_TYPE'] == 'Cash_through_the_bank').astype('i')
    df['FLAG_PAYMENT_TYPE_XNA'] = (df['NAME_PAYMENT_TYPE'] == 'XNA').astype('i')
    df.pop('NAME_PAYMENT_TYPE')

    # Days 365.243 values -> nan
    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    df['NOT_COMPLETE'] = pd.notnull(df['DAYS_LAST_DUE_1ST_VERSION']).astype('i')

    # Add flags
    df['FLAG_Approved'] = (df['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    df['FLAG_Refused'] = (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    df['FLAG_Revolving'] = (df['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')

    # Add feature: value ask / value received percentage
    df['RATIO_APP_TO_CREDIT'] = df['AMT_APPLICATION'] / (1+df['AMT_CREDIT'])

    # x-sell(discussion/63032)
    df['FLAG_X_SELL_1'] = df['NAME_PRODUCT_TYPE'].apply(has_x_sell)
    df['FLAG_X_SELL_2'] = df['PRODUCT_COMBINATION'].apply(has_x_sell)

    return df


def main():
    df = preprocess_prev()
    df.to_feather('./data/prev.preprocessed.feather')


if __name__ == '__main__':
    main()
