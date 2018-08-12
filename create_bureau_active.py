import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def aggregate_bureau():
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    df = df[df['FLAG_ACTIVE'] == 1]

    g = df.groupby('SK_ID_CURR').agg({
        # num
        'AMT_ANNUITY': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'std', 'sum', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'std', 'sum', 'min', 'max'],
        'DAYS_CREDIT_UPDATE': ['mean', 'std', 'sum', 'min', 'max'],
        # new
        'FLAG_ONGOING': ['mean'],
        # bb
        "BB_DPD_MEAN": ['mean', 'std', 'sum', 'min', 'max'],
        "BB_STATUS_NUNIQUE": ['mean', 'std', 'sum'],
    })
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['BURE_ACT_{}'.format(c) for c in g.columns]

    return g.reset_index()


def main():
    agg = aggregate_bureau()
    agg.to_feather('./data/bureau.active.feather')


if __name__ == '__main__':
    main()
