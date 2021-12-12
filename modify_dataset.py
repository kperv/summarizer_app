"""
A module to transform the dataset and prepare it for training

The dataset used is the Russian part of the MLSum dataset
https://huggingface.co/datasets/mlsum
"""

import os
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset
import pandas as pd

from clustering_model import *

DATASET = "mlsum"
DATASET_LANG = 'ru'
NUM_SLICES = 2


def make_extractive_summary(text):
    return Extractor(text).summarize()


def create_extractive_dataset(dataset):
    dataset = dataset.drop(dataset.columns[1:], axis=1)
    dataset['summary'] = ""
    dataset['summary'] = dataset.text.apply(make_extractive_summary)
    return dataset

def check_name(name):
    path = os.getcwd()
    name_path = os.path.join(path, name)
    return os.path.isfile(name_path)

def transform_and_save_batch(df, batch_num, slice_length):
    start = (batch_num - 1) * slice_length
    end = (batch_num) * slice_length
    df_slice = df[start:end]
    df_slice_transformed = create_extractive_dataset(df_slice)
    save_name = 'data/train_part_' + str(batch_num) + '.csv'
    if check_name(save_name):
        save_name = 'data/val_part_' + str(batch_num) + '.csv'
    df_slice_transformed.to_csv(save_name)
    return df_slice_transformed


def transform_dataset(df):
    transformed_df = pd.DataFrame()
    full_length = len(df)
    slice_length = full_length // NUM_SLICES
    print('Slice length is ', slice_length)
    for batch_num in range(1, NUM_SLICES+1):
        df_slice = transform_and_save_batch(df, batch_num, slice_length)
        transformed_df = pd.concat([transformed_df, df_slice])
    return transformed_df


def clean_dataset(df):
    return df.drop(df.columns[2:], axis=1)


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
        train_df = raw_datasets['train'][:]
        val_df = raw_datasets['validation'][:]
        test_df = raw_datasets['test'][:]
    return train_df, val_df, test_df


def transform(dfs):
    train_df, val_df, test_df = dfs
    train_df = transform_dataset(train_df)
    val_df = transform_dataset(val_df)
    test_df = clean_dataset(test_df)
    dfs = train_df, val_df, test_df
    return dfs

def save_modified_dataset(dfs):
    train_df, val_df, test_df = dfs
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

def form_dataset_dict(dfs):
    train_dataset, val_dataset, test_dataset = dfs
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_dataset),
        'val': Dataset.from_pandas(val_dataset),
        'test': Dataset.from_pandas(test_dataset)
    })
    return dataset


def load_and_modify_dataset(args):
    dataset = load_data(args)
    if args.use_original_dataset:
        return form_dataset_dict(dataset)
    transformed_dataset = transform(dataset)
    if args.save_modified_dataset:
        save_modified_dataset(transformed_dataset)
    dataset = form_dataset_dict(transformed_dataset)
    return dataset
