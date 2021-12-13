from rouge import Rouge
import nltk
import pandas as pd

LANG = "ru"
nltk.download('punkt')


def get_rouge_score(sample):
    rouge = Rouge()
    preprocess_exs = lambda exs : [ex.strip().lower() for ex in exs]
    predictions = list()
    predictions.append(sample['summary'])
    predictions = preprocess_exs(predictions)
    references = list()
    references.append(sample.text)
    references = preprocess_exs(references)
    predictions = [pred if len(pred) else 'а' for pred in predictions]
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    return {k: round(v['f'], 3) for k, v in rouge_scores.items()}

def add_metrics(dataset):
    dataset[['rouge-1', 'rouge-2', 'rouge-l']] = 0, 0, 0
    df = pd.DataFrame(list(dataset.apply(get_rouge_score, axis=1).values))
    dataset = df.combine_first(dataset)
    dataset = dataset.reindex(
        columns=['text', 'summary', 'rouge-1', 'rouge-2', 'rouge-l']
    )
    return dataset