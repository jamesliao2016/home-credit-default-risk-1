import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    df = pd.read_feather('./data/credit.preprocessed.feather')

    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

    grp = df.groupby(['SK_ID_CURR'])
    prev = grp.shift(-1)
    for c in prev.columns:
        if df[c].dtype == 'object':
            continue
        df[c] -= prev[c]

    grp = df.groupby('SK_ID_CURR')
    agg = grp.agg({
        'AMT_BALANCE': ['mean'],
        # 'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        # 'AMT_INST_MIN_REGULARITY': ['mean'],
        # 'AMT_PAYMENT_TOTAL_CURRENT': ['mean'],
        # 'AMT_RECEIVABLE_PRINCIPAL': ['mean'],
        # 'AMT_TOTAL_RECEIVABLE': ['mean'],
        # 'CNT_INSTALMENT_MATURE_CUM': ['mean'],
        # 'SK_DPD': ['mean'],
        # 'SK_DPD_DEF': ['mean'],
    })
    agg.columns = [a + "_" + b.upper() for a, b in agg.columns]
    agg.columns = ["CRED_DIFF" + c for c in agg.columns]
    agg = agg.reset_index()

    agg.to_feather('./data/credit.diff.feather')


if __name__ == '__main__':
    main()
