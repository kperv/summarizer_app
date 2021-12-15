"""
A module to transform the dataset and prepare it
for clustering evaluation.

The dataset used is the Russian part of the MLSum dataset
https://huggingface.co/datasets/mlsum
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
    if args.dev_mode:
        test_df = raw_datasets['train'][:25]
    else:
        test_df = raw_datasets['train'][:args.number]
    return test_df


def transform(df, args):
    df = transform_dataset(df, args)
    return df

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


def save_dataset(df):
    df.to_csv('test.csv', index=False)


def define_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Replace MLSUM 'summary' column with a generated extractive summary
                 --------------------------------
            set --dev-mode=True for fast run on 25 samples
        ''')
    )
    parser.add_argument(
        '-d',
        '--dev_mode',
        type=bool,
        default=False,
        help='run in development mode on 25 samples (default: False)'
    )
    parser.add_argument(
        '-s',
        '--slice',
        type=int,
        default=50,
        help='the length of an intermediate transformed document'
    ),
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=200,
        help='total number of rows to modify'
    )
    args = parser.parse_args()
    return args


def main():
    args = define_args()
    dataset = load_data(args)
    dataset = transform(dataset, args)
    save_dataset(dataset)


if __name__ == '__main__':
    main()