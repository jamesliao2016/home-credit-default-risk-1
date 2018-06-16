import pandas as pd
pd.set_option("display.max_columns", 100)


def preprocess_inst():
    ins_df = pd.read_feather('./data/installments_payments.csv.feather')

    # Percentage and difference paid in each installment (amount paid and installment value) # noqa
    ins_df['PAYMENT_PERC'] = ins_df['AMT_PAYMENT'] / ins_df['AMT_INSTALMENT']
    ins_df['PAYMENT_DIFF'] = ins_df['AMT_INSTALMENT'] - ins_df['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins_df['DPD'] = ins_df['DAYS_ENTRY_PAYMENT'] - ins_df['DAYS_INSTALMENT']
    ins_df['DBD'] = ins_df['DAYS_INSTALMENT'] - ins_df['DAYS_ENTRY_PAYMENT']
    ins_df['DPD'] = ins_df['DPD'].apply(lambda x: x if x > 0 else 0)
    ins_df['DBD'] = ins_df['DBD'].apply(lambda x: x if x > 0 else 0)

    return ins_df


def main():
    res = preprocess_inst()
    res.to_feather('./data/installments_payments.preprocessed.feather')


if __name__ == '__main__':
    main()
