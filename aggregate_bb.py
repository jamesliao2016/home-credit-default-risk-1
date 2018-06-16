import pandas as pd
pd.set_option("display.max_columns", 100)


def aggregate_bb():
    bb_df = pd.read_feather('./data/bureau_balance.preprocessed.feather')
    bb_df = bb_df.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])

    # first/last
    grp = bb_df.groupby('SK_ID_BUREAU')
    bb_first = grp.first()[['MONTHS_BALANCE']]
    bb_first.columns = ['FIRST_{}'.format(c) for c in bb_first.columns]
    bb_last = grp.last()[['MONTHS_BALANCE', 'STATUS']]
    bb_last.columns = ['LAST_{}'.format(c) for c in bb_last.columns]
    bb_agg = pd.concat([bb_first, bb_last], axis=1)
    bb_agg['COUNT'] = grp.size()

    # aggregate
    agg = {
        'STATUS': ['nunique'],
        'GOOD_STATUS': ['mean'],
        'BAD_STATUS': ['mean'],
    }
    grp = bb_df.groupby('SK_ID_BUREAU')
    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    bb_agg = bb_agg.join(g, on='SK_ID_BUREAU', how='left')

    bb_agg['TERM'] = bb_agg[
        'LAST_MONTHS_BALANCE'] - bb_agg['FIRST_MONTHS_BALANCE']
    bb_agg.columns = ["BB_" + c for c in bb_agg.columns]

    return bb_agg.reset_index()


def main():
    bb_agg = aggregate_bb()
    bb_agg.to_feather('./data/bureau_balance.agg.feather')


if __name__ == '__main__':
    main()
