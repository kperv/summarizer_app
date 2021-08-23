import argparse

import pandas as pd
import torch
import torch.utils.data
import pytorch_lightning as pl
import datasets
import transformers
import nltk
import json

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import load_metric
from transformers import BertTokenizer, BertModel
from transformers import DataCollatorForSeq2Seq

D = 784
MODEL = 'bert-base-multilingual-cased'
DATASET = "mlsum"

nltk.download('punkt')
metric = load_metric('bertscore')
raw_datasets = load_dataset(DATASET, "ru")


def tokenize_function(sample):
    return tokenizer(sample['text'], sample['summary'], padding=True)

def preprocess(dataset):
    dataset = dataset.remove_columns(['title', 'url', 'date', 'topic'])
    tokenizer = BertTokenizer.from_pretrained(MODEL)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['summary', 'text'])
    tokenized_dataset = tokenized_dataset.with_format('torch')
    return tokenized_dataset



small_train_dataset = raw_datasets['train'].select(range(100))
df = pd.DataFrame(small_train_dataset)

def get_texts_for_task_evaluation(dataset):
    df = pd.DataFrame(dataset['test'])
    return df['text'].sample(n=3, random_state=1).values

def get_statistics(dataset):
    df = pd.DataFrame(dataset)


df_text = df.text.str.split()
text_lens = df_text.str.len().values
print(text_lens)

df_summary = df.summary.str.split()
summary_lens = df_summary.str.len().values
print(summary_lens)



class BaseModel(pl.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(D, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, D)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


    def forward(self, x):
        z = self.encode(x.view(-1, D))
        return self.decode(z)


    def loss_function(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, D), reduction='sum')
        return BCE


    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x = self(x)
        loss = self.loss_function(recon_x, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x = self(x)
        val_loss = self.loss_function(recon_x, x)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


    def train_dataloader(self):
        return DataLoader(
            tokenized_datasets['train'],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator
        )


    def val_dataloader(self):
        val_loader = DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size)
        return val_loader

def save_args():
    with open('args.json', 'w') as f:
        json.dump(args, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="New model")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    save_args()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = BaseModel()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.save_checkpoint("could_be_a_summarizer.ckpt")
