"""
A module to transform the dataset and prepare it for training

The dataset used is the Russian part of the MLSum dataset
https://huggingface.co/datasets/mlsum
Due to time limits the full dataset is cut
and only the 1/5 of it is taken for transformation.
"""


import os
import argparse
import textwrap

from datasets import load_dataset
import pandas as pd

from clustering_model import *


DATASET = "mlsum"
DATASET_LANG = 'ru'


def load_data(args):
    raw_datasets = load_dataset(DATASET, DATASET_LANG)
    raw_datasets['train'].set_format("pandas")
    raw_datasets['validation'].set_format("pandas")
    raw_datasets['test'].set_format("pandas")
    if args.dev_mode:
        train_df = raw_datasets['train'][:25]
        val_df = raw_datasets['validation'][:5]
        test_df = raw_datasets['test'][:5]
    else:
        train_df = raw_datasets['train'][:5000]
        val_df = raw_datasets['validation'][:100]
        test_df = raw_datasets['test'][:100]
    return train_df, val_df, test_df


def transform(dfs, args):
    train_df, val_df, test_df = dfs
    train_df = transform_dataset(train_df, args)
    val_df = transform_dataset(val_df, args)
    test_df = transform_dataset(test_df, args)
    dfs = train_df, val_df, test_df
    return dfs

def transform_dataset(df, args):
    full_length = len(df)
    num_slices = full_length // args.slice
    for batch_num in range(1, num_slices+1):
        transform_and_save_batch(df, batch_num, args)
    transformed_df = collect_modified_dataset()
    return transformed_df

def transform_and_save_batch(df, batch_num, args):
    start = (batch_num-1) * args.slice
    end = (batch_num) * args.slice
    df_slice = df[start:end]
    df_slice_transformed = create_extractive_dataset(df_slice)
    save_name = 'modified_MLSUM_part_' + str(batch_num) + '.csv'
    save_path = os.path.join(os.getcwd(), save_name)
    df_slice_transformed.to_csv(save_path)
    return df_slice_transformed


def create_extractive_dataset(dataset):
    dataset = dataset.drop(dataset.columns[1:], axis=1)
    dataset['summary'] = ""
    dataset['summary'] = dataset.text.apply(make_extractive_summary)
    return dataset


def make_extractive_summary(text):
    return Extractor(text).summarize()


def collect_modified_dataset():
    transformed_df = pd.DataFrame()
    data_folder = os.path.join(os.getcwd(), '/data')
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    for dataset_part in os.listdir(data_folder):
        if 'modified_MLSUM_part_' in dataset_part:
            part_path = os.path.join(data_folder, dataset_part)
            df_slice = pd.read_csv(part_path)
            transformed_df = pd.concat([transformed_df, df_slice])
            os.remove(dataset_part)
    return transformed_df


def save_dataset(dfs):
    train_df, val_df, test_df = dfs
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)


def define_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Replace MLSUM 'summary' column with a generated extractive summary
                 --------------------------------
            set --dev-mode=True for fast run on 25,5,5 samples
        ''')
    )
    parser.add_argument(
        '-d',
        '--dev_mode',
        type=bool,
        default=False,
        help='run in development mode on (25,5,5) samples (default: False)'
    )
    parser.add_argument(
        '-s',
        '--slice',
        type=int,
        default=100,
        help='the length of an intermediate transformed document'
    ),
    parser.add_argument(
        '-o',
        '--original',
        type=bool,
        default=False,
        help='use MLSUM for training without modifications'
    )
    args = parser.parse_args()
    return args


def main():
    args = define_args()
    dataset = load_data(args)
    if not args.original:
        dataset = transform(dataset, args)
    save_dataset(dataset)


if __name__ == '__main__':
    main()