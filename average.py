import os
import pandas as pd
from datetime import datetime
now = datetime.now().strftime('%m%d-%H%M')


def main():
    n = 1
    basename = './output/0825-1728.0.csv.gz'
    print(basename)
    df = pd.read_csv(basename)
    df = df.set_index('SK_ID_CURR')
    for fname in [
        './output/0814-2333.csv.gz',  # 0.7953235351646931 +- 0.00202180190272089
        './output/0825-2200.1.csv.gz',  # 0.7973598888313274 +- 0.001968567914083943
        './output/0826-0025.2.csv.gz',  # 0.7970876788137937 +- 0.001964420220658776
        './output/0826-0922.3.csv.gz',  # 0.7972619179046415 +- 0.0018625013852117517
    ]:
        print(fname)
        tmp = pd.read_csv(fname)
        tmp = tmp.set_index('SK_ID_CURR')
        df['TARGET'] += tmp['TARGET']
        n += 1

    df['TARGET'] /= n
    df = df.reset_index()
    path = os.path.join('./output', '{}.avg.csv.gz'.format(now))
    print('save {}'.format(path))
    df.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
