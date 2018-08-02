import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def preprocess_prev():
    df = pd.read_feather('./data/previous_application.feather')

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
    return df


def main():
    df = preprocess_prev()
    df.to_feather('./data/prev.preprocessed.feather')


if __name__ == '__main__':
    main()
