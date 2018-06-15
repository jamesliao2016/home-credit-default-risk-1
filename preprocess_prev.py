import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def preprocess_prev():
    pre_df = pd.read_feather('./data/previous_application.feather')
    pre_df = pre_df.sort_values(['SK_ID_CURR', 'SK_ID_PREV'])

    # Days 365.243 values -> nan
    pre_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    pre_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    pre_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    pre_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    pre_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    pre_df['RATIO_APP_TO_CREDIT'] =\
        pre_df['AMT_APPLICATION'] / pre_df['AMT_CREDIT']
    return pre_df


def main():
    pre_df = preprocess_prev()
    pre_df.to_feather('./data/previous_application.preprocessed.feather')


if __name__ == '__main__':
    main()
