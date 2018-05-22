import pandas as pd


def convert(src):
    print('convert {}'.format(src))
    dst = src.replace('zip', 'feather')
    df = pd.read_csv(src)
    df.to_feather(dst)


def main():
    convert('./data/POS_CASH_balance.csv.zip')
    convert('./data/application_train.csv.zip')
    convert('./data/application_test.csv.zip')
    convert('./data/bureau.csv.zip')
    convert('./data/bureau_balance.csv.zip')
    convert('./data/credit_card_balance.csv.zip')
    convert('./data/installments_payments.csv.zip')
    convert('./data/previous_application.csv.zip')
    convert('./data/sample_submission.csv.zip')


if __name__ == '__main__':
    main()
