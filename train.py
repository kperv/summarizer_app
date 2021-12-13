import os
import argparse
import textwrap
from collections import OrderedDict
import torch
from rouge import Rouge
import nltk
import numpy as np
import pandas as pd
from datasets import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import metrics as m


MODEL_CHECKPOINT = "t5-small"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

def define_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            '''\
                Training a summarizing model
                --------------------------------
                set --dev-mode=True for fast run on 25,5,5 samples
            ''')
        )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=2,
        help='input batch size for training (default: 2)'
    ),
    parser.add_argument(
        '-e',
        '--epochs',
        type=int, default=1,
        help='number of epochs to train (default: 1)'
    ),
    parser.add_argument(
        '-sm',
        '--save_model',
        type=bool,
        default=False,
        help='save a pretrained model in the project folder'
    )
    args = parser.parse_args()
    args.device = torch.cuda.is_available()
    return args


def read_dataset():
    train = pd.read_csv('data/train.csv')
    val = pd.read_csv('data/val.csv')
    test = pd.read_csv('data/test.csv')
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train),
        'val': Dataset.from_pandas(val),
        'test': Dataset.from_pandas(test)
    })
    return dataset


def train(dataset, args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def compute_metrics(eval_pred):
        rouge = Rouge()
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_preds = [nltk.sent_tokenize(pred.strip()) for pred in decoded_preds]
        decoded_preds = [pred if len(pred) else 'а' for pred in decoded_preds]
        decoded_preds = ["\n".join(pred) for pred in decoded_preds]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [nltk.sent_tokenize(label.strip()) for label in decoded_labels]
        decoded_labels = [label if len(label) else 'а' for label in decoded_labels]
        decoded_labels = ["\n".join(label) for label in decoded_labels]

        result = rouge.get_scores(hyps=decoded_preds, refs=decoded_labels, avg=True)
        result = {key: value['f'] * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    if args.device:
        model.to(DEVICE)

    model_name = MODEL_CHECKPOINT.split("/")[-1]
    logging_steps = len(dataset["train"]) // (args.batch_size * args.epochs)

    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-mlsum-ru",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=True,
        logging_steps=logging_steps
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    if args.save_model:
        model.save_pretrained('.')
    predictions = trainer.predict(tokenized_datasets["test"])
    return tokenizer, predictions


def save_predictions(dataset, tokenizer, predictions):
    summaries = tokenizer.batch_decode(
        predictions.predictions,
        skip_special_tokens=True
    )
    dataset['test'].set_format('pandas')
    result_df = dataset['test'][:]
    result_df['summary'] = summaries
    return result_df


def calculate_metrics(result_df):
    result_df = m.add_metrics(result_df)
    result_df.to_csv('predictions.csv')
    metrics = OrderedDict({
        'rouge-1': round(result_df['rouge-1'].mean(), 3),
        'rouge-2': round(result_df['rouge-2'].mean(), 3),
        'rouge-l': round(result_df['rouge-l'].mean(), 3)
    })
    return metrics


def save_metrics_to_file(metrics, report_file="metrics.txt"):
    report_path = os.getcwd() + "/" + report_file
    if os.path.isfile(report_path):
        os.remove(report_path)
    with open(report_file, "a") as report:
        report.write("\n")
        intro_str = ("Summarization model" + "\n")
        report.write(intro_str)
        report.write("\n")
        report.write("-" * 30)
        report.write("\n")
        report.write("\n")
        for name, value in metrics.items():
            report.write(name + ': ' + str(value) + '\n')
            report.write("\n")

def main():
    args = define_args()
    dataset = read_dataset()
    preds = train(dataset, args)
    df = save_predictions(dataset, *preds)
    metrics = calculate_metrics(df)
    save_metrics_to_file(metrics)


if __name__ == '__main__':
    main()