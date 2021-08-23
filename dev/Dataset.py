import torch
import torch.utils.data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset

import datasets
import transformers
import nltk

from datasets import load_dataset
from datasets import load_metric
from transformers import BertTokenizer, BertModel

CHECKPOINT = 'bert-base-multilingual-cased'
DATASET = "mlsum"

nltk.download('punkt')
metric = load_metric('bertscore')

class MlSumDataModule(pl.LightningDataModule):
  def __init__(self, batch_size=32):
    super.__init__()
    self.batch_size = batch_size

  def prepare_data(self):
    self.dataset_ru = load_dataset("mlsum", "ru")
    self.dataset_es = load_dataset("mlsum", "es")
    nlp_ru = spacy.load("ru_core_news_md")
    nlp_es = spacy.load("es_core_news_md")

  def train_dataloader(self):
    dataset_ru_train = DataLoader(
        self.dataset_ru["train"],
        batch_size=self.batch_size)
    dataset_es_train = DataLoader(
        self.dataset_es["train"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_train, dataset_es_train]
    return train_loaders

  def val_dataloader(self):
    dataset_ru_val = DataLoader(
        self.dataset_ru["val"],
        batch_size=self.batch_size)
    dataset_es_val = DataLoader(
        self.dataset_es["val"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_val, dataset_es_val]
    return val_loaders

  def test_dataloader(self):
    dataset_ru_test = DataLoader(
        self.dataset_ru["test"],
        batch_size=self.batch_size)
    dataset_es_test = DataLoader(
        self.dataset_es["test"],
        batch_size=self.batch_size)
    loaders = [dataset_ru_test, dataset_es_test]
    return test_loaders

raw_dataset = load_dataset(DATASET, "ru")

tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)

def tokenize_function(sample):
    return tokenizer(sample['text'], sample['summary'], padding=True, truncation=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

print(tokenized_datasets.column_names)

##### Metric
text = raw_dataset['train'][0]['text']
sentences = nltk.tokenize.sent_tokenize(text)
summary = raw_dataset['train'][0]['summary']

references = [sentences]
predictions = [summary]
P = metric.compute(predictions=predictions, references=references, lang='ru')
