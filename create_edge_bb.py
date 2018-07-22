import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)


def main():
    bb = pd.read_feather('./data/bureau_balance.preprocessed.feather')
    bb = bb.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])

    grp = bb.groupby(['SK_ID_BUREAU'])
    last = grp.last()
    last.columns = ['LAST_{}'.format(c) for c in last.columns]
    first = grp.first()
    first.columns = ['FIRST_{}'.format(c) for c in first.columns]

    edge = last.join(first, on='SK_ID_BUREAU')
    edge['TERM'] = edge['LAST_MONTHS_BALANCE'] - edge['FIRST_MONTHS_BALANCE']
    edge = edge.reset_index()

    edge.to_feather('./data/bb.edge.feather')


if __name__ == '__main__':
    main()
