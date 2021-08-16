"""
Implementation of text extractive summarizer with conditional output.

Text written in two languages can be processed and summarized: Russian and Spanish.
Words tokenized and encoded by Bert pretrained model, 
sentence embeddings got as a mean of token embeddings in a sentence. 
afterwards sentece embeddings are clustered into defined number of groupds.
Closest sentences into each group are chosen for resulting summary.
"""


import numpy as np
import spacy
import transformers

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from bert_score import score


MODEL_NAME = 'bert-base-multilingual-cased'


class Summarizer():
    """ Process text and make a summary with the set length."""
    
    def __init__(self, text, number, language):
        self.text = text
        self.number = number
        self.lang = language
        self.nlp = _get_spacy_object()

    def _get_spacy_object(self):
        """Download a spacy pipeline object for text processing"""
        try:
            if self.lang == "ru":
                nlp = spacy.load("ru_core_news_md")
            elif self.lang == "es":
                nlp = spacy.load("es_core_news_md")
            else:
                raise ValueError
        except ValueError:
            print('Incorrect language.')
        except:
            print("Unable to load spacy object. Got an error: {}".format(sys.exc_info()[0]))
        else:
            return nlp

    def break_text_into_sentences(self):
        """Transform a text into a list of sentences"""
        doc = self.nlp(self.text)
        assert doc.has_annotation("SENT_START")
        sentences = [str(sent) for sent in doc.sents]
        return sentences

    def tokenize_sentences(self, sentences):
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        tokenized_sentences = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")
        return tokenized_sentences

    def encode_sentences(self, tokenized_sentences):
        model = BertModel.from_pretrained(MODEL_NAME)
        outputs = model(tokenized_sentences.input_ids)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings.detach().numpy()
        return embeddings

    def get_sentence_embeddings(self, embeddings):
        return embeddings.mean(axis=1)

    def cluster_sentence_embeddings(self, sentence_embeddings):
        kmeans = KMeans(n_clusters=self.number).fit(sentence_embeddings)
        centroids = kmeans.cluster_centers_
        return centroids

    def get_n_closest(self, sentence_embeddings, centroids):
        distances, min_distance_positions = pairwise_distances_argmin_min(
            sentence_embeddings, 
            centroids
            )
        positions = list(np.argsort(min_distance_positions)[:number])
        return positions

    def collect_summary(self, sentences, positions):
        summary = ""
        for idx in positions:
            summary += sentences[idx]
        return summary

    def get_score(summary):
        predictions = [summary]
        references = [self.text]
        P, R, F1 = score(predictions, references, lang=self.lang)
        return F1.mean()

    def summarize(self):
        sent = break_text_into_sentences()
        tok_sent = tokenize_sentences(sent)
        emb = encode_sentences(tok_sent)
        semb = get_sentence_embeddings(emb)
        cent = cluster_sentence_embeddings(semb)
        pos = get_n_closest(semb, cent)
        summary = collect_summary(sent, pos)
        score = get_score(summary)
        return summary, score








    
    
    