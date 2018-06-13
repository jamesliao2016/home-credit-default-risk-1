import pandas as pd
pd.set_option("display.max_columns", 100)


def aggregate_bb():
    bb_df = pd.read_feather('./data/bureau_balance.preprocessed.feather')
    agg = {'MONTHS_BALANCE': ['min', 'max', 'count']}
    for c in bb_df.columns:
        if c == 'SK_ID_BUREAU' or c in agg:
            continue
        agg[c] = ['mean']
    bb_agg = bb_df.groupby('SK_ID_BUREAU').agg(agg)
    bb_agg.columns = [
        a + "_" + b.upper() for a, b in bb_agg.columns]
    bb_agg['TERM'] = bb_agg[
        'MONTHS_BALANCE_MAX'] - bb_agg['MONTHS_BALANCE_MIN']
    bb_agg.columns = ["BB_" + c for c in bb_agg.columns]

    return bb_agg.reset_index()


def main():
    bb_agg = aggregate_bb()
    bb_agg.to_feather('./data/bureau_balance.agg.feather')


if __name__ == '__main__':
    main()
