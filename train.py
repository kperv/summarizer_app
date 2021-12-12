"""A module for training the model with the dataset"""


import textwrap
import os
import argparse
from collections import OrderedDict
import torch
from rouge import Rouge
import nltk
import numpy as np
import pandas as pd
from datasets import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import metrics as m
import modify_dataset as md


model_checkpoint = "t5-small"
max_input_length = 1024
max_target_length = 128
if torch.cuda.is_available():
    device = torch.device("cuda")

def define_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                 Training a summarizing model
                 --------------------------------
                 to run on 30 samples set --dev-mode=True


            ''')
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=2,
        help='input batch size for training (default: 2)'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int, default=1,
        help='number of epochs to train (default: 1)'
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
        '--save_modified_dataset',
        type=bool,
        default=False,
        help='save modified MLSUM dataset to (train,val,test) .csv files'
    ),
    parser.add_argument(
        '-m',
        '--modify_dataset',
        type=bool,
        default=False,
        help='to modify MLSUM dataset to contain extractive summaries'
    )
    parser.add_argument(
        '-sm',
        '--save_model',
        type=bool,
        default=False,
        help='save pretrained model weights in the project folder'
    )
    parser.add_argument(
        '-o',
        '--use_original_dataset',
        type=bool,
        default=False,
        help='use MLSUM for training without modifications'
    )

    args = parser.parse_args()
    args.device = torch.cuda.is_available()
    return args


def get_dataset(args):
    if args.modify_dataset or args.use_original_dataset:
        dataset = md.load_and_modify_dataset(args)
    else:
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
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def compute_metrics(eval_pred):
        rouge = Rouge()
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = rouge.get_scores(hyps=decoded_preds, refs=decoded_labels, avg=True)
        # Extract a few results
        result = {key: value['f'] * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    if args.device:
        model.to(device)

    model_name = model_checkpoint.split("/")[-1]
    logging_steps = len(dataset["train"]) // (args.batch_size * args.epochs)

    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-mlsum-ru",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
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
    metrics = OrderedDict({
        'Bert-score': result_df.bert_score.mean(),
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
    dataset = get_dataset(args)
    preds = train(dataset, args)
    df = save_predictions(dataset, *preds)
    metrics = calculate_metrics(df)
    save_metrics_to_file(metrics)


if __name__ == '__main__':
    main()