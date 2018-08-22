import pandas as pd


def load_train(idx):
    print('load train by fold: {}'.format(idx))
    df = pd.read_feather('./data/features.normalized.feather')
    fold = pd.read_feather('./data/fold.{}.feather'.format(idx))
    feat = pd.read_feather('./data/nn.fold.{}.feather'.format(idx))

    df['nn_0'] = feat[' nn_0']
    df = df[['SK_ID_CURR', 'nn_0']]
    df = df[df['SK_ID_CURR'].isin(fold['SK_ID_CURR'])]
    return df.set_index('SK_ID_CURR')


def load_test(idx):
    print('load test by fold: {}'.format(idx))
    df = pd.read_feather('./data/features.normalized.feather')
    feat = pd.read_feather('./data/nn.fold.{}.feather'.format(idx))

    df['nn_0'] = feat[' nn_0']
    df = df[pd.isnull(df['TARGET'])]
    df = df[['SK_ID_CURR', 'nn_0']]
    df['nn_0'] /= 5
    return df.set_index('SK_ID_CURR')


def main():
    concat = []
    test = None
    for i in range(5):
        train = load_train(i)
        concat.append(train)
        if test is None:
            test = load_test(i)
        else:
            test += load_test(i)

    concat.append(test)
    df = pd.concat(concat).reset_index()
    df.to_feather('./data/nn.feather')


if __name__ == '__main__':
    main()
