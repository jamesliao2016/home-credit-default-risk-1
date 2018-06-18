def aggregate_inst(df, key):
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
    agg = g

    agg['COUNT'] = grp.size()

    return agg
