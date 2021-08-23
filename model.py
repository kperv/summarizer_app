"""
A module for Bart fine-tuning for the summarization task
according to the Pytorch lightning structure.
"""

__all__ = ["SumModel"]


import pytorch_lightning as pl
from transformers import BartTokenizer, BartForConditionalGeneration
from torch import optim


MODEL_NAME = BartForConditionalGeneration
CHECKPOINT = 'facebook/bart-base'
TOKENIZER = BartTokenizer


class SumModel(pl.LightningModule):
  def __init__(self, model_name, checkpoint_name, learning_rate):
    super().__init__()
    self.model = model_name.from_pretrained(checkpoint_name)
    self.learning_rate = learning_rate

  def forward(self, input_ids, labels):
    loss, _ = self.model(input_ids, labels)
    return loss

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    labels = batch['labels']
    loss = self(input_ids=input_ids, labels=labels)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    labels = batch['labels']
    loss = self(input_ids=input_ids, labels=labels)
    self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return {'val_loss': loss}

  def test_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    labels = batch['labels']
    loss = self(input_ids=input_ids, labels=labels)
    return {'test_loss': loss}

  def configure_optimizers(self):
      return optim.AdamW(self.parameters(), lr=self.learning_rate)