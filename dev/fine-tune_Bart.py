import datasets
import transformers
import nltk
import rouge-score

nltk.download('punkt')

from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration

CHECKPOINT = 'facebook/bart-base'
DATASET = "mlsum"

raw_dataset = load_dataset(DATASET, "ru")

tokenizer = BartTokenizer.from_pretrained(CHECKPOINT)
model = BartForConditionalGeneration.from_pretrained(CHECKPOINT)

train_dataset = raw_dataset['train'].select(range(32))
val_dataset = raw_dataset['validation'].select(range(32))

max_input_length = 512
max_target_length = 64

def tokenize_function(sample):
    model_inputs = tokenizer(sample['text'], max_length=max_input_length, truncation=True)
    labels = tokenizer(sample['summary'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(tokenize_function)
val_dataset = val_dataset.map(tokenize_function)

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

batch_size = 8

args = Seq2SeqTrainingArguments(
    "test-summarization",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)

from datasets import load_metric

metric = load_metric("rouge")

import nltk
import numpy as np

nltk.download('punkt')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

ARTICLE_TO_SUMMARIZE = 'Платой за проделанную работу становились… кошельки, сумки и мобильные телефоны. Как удалось узнать “МК”, 57-летний ловкач провел в местах лишения свободы 30 лет своей жизни. На свободе он сразу же брался за старое. Отбыв последний срок, рецидивист, не мудрствуя лукаво, вновь приступил к ремеслу. В последний раз мужчина разработал оригинальный сценарий. Раздобыв адреса пенсионеров, вор отправлялся “на дело”: звонил в квартиры, представлялся электриком из ЖЭУ и говорил, что ему необходимо проверить проводку. Доверчивые граждане впускали “электрика”. Негодяй присматривал ридикюли и кошельки граждан и прятал их в свою рабочую сумку. После этого откланивался и был таков. А пенсионеры, обнаружив пропажу, бежали в милицию. Стражи порядка задержали рецидивиста через несколько дней после его последнего “выступления”, когда он пытался уехать в Ярославль.'
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print(summary_ids)
summary = tokenizer.decode(summary_ids)
print(summary)

model.save_pretrained()