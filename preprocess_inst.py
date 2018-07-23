import pandas as pd
from utility import reduce_memory
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


def preprocess_inst():
    ins = pd.read_feather('./data/installments_payments.feather')
    ins = ins.sort_values(
        ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT'])

    ins['IS_CREDIT'] = (ins['NUM_INSTALMENT_VERSION'] == 0).astype('i')

    # Percentage and difference paid in each installment (amount paid and installment value) # noqa
    ins['RATIO_PAYMENT'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['DIFF_PAYMENT'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['FLAG_DIFF_PAYMENT'] = (ins['DIFF_PAYMENT'] > 0).astype('int8')
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['FLAG_DPD'] = (ins['DPD'] > 0).astype('int8')
    ins['FLAG_DBD'] = (ins['DBD'] > 0).astype('int8')

    reduce_memory(ins)

    return ins.reset_index(drop=True)


def main():
    res = preprocess_inst()
    res.to_feather('./data/inst.preprocessed.feather')


if __name__ == '__main__':
    main()
