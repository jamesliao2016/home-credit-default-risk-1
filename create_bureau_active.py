import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def aggregate_bureau():
    df = pd.read_feather('./data/bureau.preprocessed.feather')
    df = df[df['FLAG_ACTIVE'] == 1]

    g = df.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['mean', 'std', 'sum', 'min', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'std', 'sum', 'min', 'max'],
    })
    g.columns = [x + "_" + y.upper() for x, y in g.columns]
    g.columns = ['BURE_ACT_{}'.format(c) for c in g.columns]

    return g.reset_index()


def main():
    agg = aggregate_bureau()
    agg.to_feather('./data/bureau.active.feather')


if __name__ == '__main__':
    main()
