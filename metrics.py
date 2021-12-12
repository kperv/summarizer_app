from rouge import Rouge
from bert_score import score
import nltk
import pandas as pd

LANG = "ru"
nltk.download('punkt')


def get_bert_score(sample):
    predictions = []
    predictions.append(sample['summary'])
    references = []
    references.append(sample.text)
    P, R, F1 = score(predictions, references, lang=LANG)
    bert_score = F1.mean().numpy()
    return bert_score

def get_rouge_score(sample):
    rouge = Rouge()
    predictions = []
    predictions.append(sample['summary'])
    references = []
    references.append(sample.text)
    preprocess_exs = lambda exs : [ex.strip().lower() for ex in exs]
    rouge_scores = rouge.get_scores(
        preprocess_exs(predictions),
        preprocess_exs(references),
        avg=True
    )
    return {k: round(v['f'], 3) for k, v in rouge_scores.items()}

def add_metrics(dataset):
    dataset = dataset.drop(dataset.columns[2:], axis=1)
    dataset['bert_score'] = ""
    dataset['bert_score'] = dataset.apply(get_bert_score, axis=1)
    dataset[['rouge-1', 'rouge-2', 'rouge-l']] = 0, 0, 0
    df = pd.DataFrame(list(dataset.apply(get_rouge_score, axis=1).values))
    dataset = df.combine_first(dataset)
    dataset = dataset.reindex(
        columns=['text', 'summary', 'bert_score', 'rouge-1', 'rouge-2', 'rouge-l']
    )
    return dataset