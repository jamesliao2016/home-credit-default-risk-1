import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def aggregate_inst():
    df = pd.read_feather('./data/installments_payments.preprocessed.feather')
    df = df.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])

    # TODO: Move to preprocess
    df['IS_CREDIT'] = (df['NUM_INSTALMENT_VERSION'] == 0).astype('i')

    key = ['SK_ID_CURR', 'SK_ID_PREV']
    # Features: Perform aggregations
    agg = {
        # original
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'min'],
        # preprocessed
        'RATIO_PAYMENT': ['max', 'min', 'mean', 'std'],
        'DIFF_PAYMENT': ['max', 'min', 'mean', 'sum', 'std'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'IS_CREDIT': ['mean'],
        'TSDIFF_DAYS_INSTALMENT': ['mean', 'std'],
        'TSDIFF_DAYS_ENTRY_PAYMENT': ['mean', 'std'],
        'TSDIFF_AMT_INSTALMENT': ['mean', 'std'],
        'TSDIFF_AMT_PAYMENT': ['mean', 'std'],
    }

    grp = df.groupby(key)
    g = grp.agg(agg)
    g.columns = [a + "_" + b.upper() for a, b in g.columns]
    ins_agg = g

    # Count installments accounts
    ins_agg['COUNT'] = grp.size()
    ins_agg.columns = ["INS_" + c for c in ins_agg.columns]
    return ins_agg.reset_index()


def main():
    agg = aggregate_inst()
    agg.to_feather('./data/installments_payments.agg.feather')


if __name__ == '__main__':
    main()
