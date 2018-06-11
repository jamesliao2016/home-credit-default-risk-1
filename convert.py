import argparse
import pandas as pd


def convert(src, dst):
    print('convert {} to {}'.format(src, dst))
    df = pd.read_csv(src)
    df.to_feather(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
