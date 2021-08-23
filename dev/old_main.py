import argparse
import sys

from clustering_model.py import *

def set_args():
    parser = argparse.ArgumentParser(description='Summarize text into n sentences')
    parser.add_argument('text', help='Text file with a text to summarize')
    parser.add_argument('number', help='Number of sentences in a summary')
    parser.add_argument('language', help='Language of the text: (ru, es)')
    args = parser.parse_args()

def save_summary(summary):
    with open('summary.txt', 'w') as file:
        file.write(summary)

if __name__ == '__main__':
    text, number, language = sys.argv[1], int(sys.argv[1]), sys.argv[2]
    summarizer = Summarizer(text, number, language)
    summary, score = summarizer.summarize()
    print(summary)
    print(score)

