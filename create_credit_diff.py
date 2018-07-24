import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    cred = pd.read_feather('./data/credit_card_balance.preprocessed.feather')

    cred = cred.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    grp = cred.groupby('SK_ID_CURR')
    cred['AMT_BALANCE_DIFF'] = grp['AMT_BALANCE'].diff()
    grp = cred.groupby('SK_ID_CURR')

    agg = grp.agg({
        'AMT_BALANCE_DIFF': ['mean'],
    })
    agg.columns = [a + "_" + b.upper() for a, b in agg.columns]
    agg.columns = ["CRED_" + c for c in agg.columns]
    agg = agg.reset_index()

    agg.to_feather('./data/credit.agg.diff.feather')


if __name__ == '__main__':
    main()
