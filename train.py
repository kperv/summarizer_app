"""A module for training the model with the dataset"""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from transformers import BartTokenizer, BartForConditionalGeneration

from model import *
from data_preprocessing import *

pl.seed_everything(42)

MODEL_NAME = BartForConditionalGeneration
CHECKPOINT_NAME = 'facebook/bart-base'
TOKENIZER_NAME = BartTokenizer
LEARNING_RATE = 0.001

def train():
    parser = argparse.ArgumentParser(description="New model")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 128')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    data_module = SumDataModule(
        batch_size=args.batch_size,
        tokenizer_name=TOKENIZER_NAME,
        checkpoint_name=CHECKPOINT_NAME
    )
    sum_model = SumModel(
        model_name=MODEL_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        learning_rate=LEARNING_RATE
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_checkpoint",
        monitor="val_loss",
        mode="min",
        verbose=True
    )
    logger = pl_loggers.TensorBoardLogger("lightning_logs", name="summarization_task")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(sum_model, data_module)
    trainer.test(sum_model, test_dataloaders=data_module.val_dataloader)
    trained_model = SumModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return trained_model


if __name__ == '__main__':
    model = train()