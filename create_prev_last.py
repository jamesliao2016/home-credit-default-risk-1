import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def main():
    prev = pd.read_feather('./data/previous_application.preprocessed.feather')
    prev = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

    last = prev.groupby(['SK_ID_CURR']).last()
    last.columns = ['PREVLAST_{}'.format(c) for c in last.columns]
    last = last.reset_index()

    last.to_feather('./data/previous_application.last.feather')


if __name__ == '__main__':
    main()
