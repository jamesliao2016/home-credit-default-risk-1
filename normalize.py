import pandas as pd


def main():
    df = pd.read_feather('./data/features.feather')
    for c in df.columns:
        if c in ['SK_ID_CURR', 'TARGET']:
            continue
        if str(df[c].dtype) == 'category':
            continue
        m = df[c].mean()
        df[c] -= m
        df[c].fillna(0, inplace=True)
        s = df[c].std()
        if pd.isnull(s) or abs(s) < 1e-6:
            print(c, s)
            df.pop(c)
        else:
            df[c] /= s
    df.to_feather('./data/features.normalized.feather')


if __name__ == '__main__':
    main()
