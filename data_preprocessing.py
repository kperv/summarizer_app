"""
A module to load and transform the dataset and prepare it for training

The dataset used is MLSum https://huggingface.co/datasets/mlsum
For training the 'text' and 'summary' columns of the dataset are used
for tokenization with BartTokenizer into a lists of tokens.
The tokenized 'summary' renamed and put into a column 'labels'
as for the summarization task the model (Bart Model for Conditioned Generation)
expects input_ids and labels, apart from other optional arguments.
"""

__all__ = ["SumDataModule"]

import pandas as pd
import torch
import pytorch_lightning as pl

from transformers import BartTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


DATASET = "mlsum"
LANG ='ru'
CHECKPOINT_NAME = 'facebook/bart-base'
TOKENIZER_NAME = BartTokenizer
BATCH_SIZE = 8


class SumDataModule(pl.LightningDataModule):
    """ Preprocess of the dataset for training"""

    def __init__(self, batch_size, tokenizer_name, checkpoint_name):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.checkpoint_name = checkpoint_name
        self.dataset = self.prepare_data()

    def _tokenize_function(self, sample):
        tokenizer = self.tokenizer_name.from_pretrained(self.checkpoint_name)
        model_inputs = tokenizer(sample['text'], truncation=True)
        labels = tokenizer(sample['summary'], truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def get_n_raw_texts(self, n=3):
        raw_dataset = load_dataset(DATASET, LANG)
        df = pd.DataFrame(raw_dataset['test'])
        return df['text'].sample(n, random_state=1).values

    def prepare_data(self):
        raw_dataset = load_dataset(DATASET, LANG)
        tokenized_dataset = raw_dataset.map(self._tokenize_function)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return tokenized_dataset

    def setup(self, stage):
        train_dataset = self.dataset['train']
        val_dataset = self.dataset['val']
        test_dataset = self.dataset['test']

    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size
        )
