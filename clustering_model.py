"""
Implementation of an extractive summarizer.

The model performs KMeans clusterisation of Bert embeddings
into clusters. Closest sentences to a centroid in each group
are chosen to form a summary.
"""


__all__ = ["Extractor"]


import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


MODEL_NAME = 'bert-base-multilingual-cased'
NUM_SENTENCES = 3
LANG = "ru"
nltk.download('punkt')


class Extractor():
    """ Process text and make a summary from the closest sentences
    to the meaning of the text."""
    def __init__(self, text, n_sentences=NUM_SENTENCES):
        super(Extractor, self).__init__()
        self.text = text
        self.n_sentences = n_sentences
        self.sentences = self.break_text_into_sentences()
        self.tokenized_sentences = self.tokenize_sentences()
        self.sentence_embeddings = self.encode_sentences()
        self.centroids = self.cluster_sentence_embeddings()
        self.positions = self.get_n_closest()

    def break_text_into_sentences(self):
        """Transform a text into a list of sentences"""
        return nltk.tokenize.sent_tokenize(self.text)

    def tokenize_sentences(self):
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        return tokenizer(
            self.sentences,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

    def encode_sentences(self):
        model = BertModel.from_pretrained(MODEL_NAME)
        outputs = model(self.tokenized_sentences.input_ids)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings.detach().numpy()
        return embeddings.mean(axis=1)

    def cluster_sentence_embeddings(self):
        kmeans = KMeans(n_clusters=self.n_sentences).fit(self.sentence_embeddings)
        return kmeans.cluster_centers_

    def get_n_closest(self):
        distances, min_distance_positions = pairwise_distances_argmin_min(
            self.sentence_embeddings,
            self.centroids
            )
        return list(np.argsort(min_distance_positions)[:self.n_sentences])

    def extract_summary(self):
        summary = ""
        for idx in self.positions:
            summary += self.sentences[idx]
            summary += ' '
        return summary

    def summarize(self):
        return self.extract_summary()
