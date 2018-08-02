import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def main():
    prev = pd.read_feather('./data/prev.preprocessed.feather')
    prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

    last = prev.groupby(['SK_ID_CURR']).last()
    last = last.drop(['SK_ID_PREV'], axis=1)
    last.columns = ['PREV_LAST_{}'.format(c) for c in last.columns]
    last = last.reset_index()

    last.to_feather('./data/prev.last.feather')


if __name__ == '__main__':
    main()
