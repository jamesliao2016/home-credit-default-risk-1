import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    pos = pd.read_feather('./data/pos.preprocessed.feather')
    pos = pos.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

    grp = pos.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
    last = grp.last()
    last.columns = ['LAST_{}'.format(c) for c in last.columns]
    first = grp.first()
    first.columns = ['FIRST_{}'.format(c) for c in first.columns]

    edge = last.join(first, on=['SK_ID_CURR', 'SK_ID_PREV'])
    edge = edge.reset_index()

    edge.to_feather('./data/pos.edge.feather')


if __name__ == '__main__':
    main()
