## A web application for extractive summarization 


**Sentence embeddings clustering for text summarization.**

This is a project for Natural Language Processing course (ods.ai, Huawei, december 2021)

Full project is presented in the Colab notebook Project.ipynb

### Introduction

There are two ways to create a summary of a text: abstractive and extractive summarization. 

Extractive summarization can be created by extracting most relevant parts of the text and combining them together.

Abstractive summarization is a Language Model generated text. For example, there are several language models, recently fine-tuned on Russian new datasets for summarization:

Model | ROUGE-1 | ROUGE-L
---|---|---
`ria_copynet_10kk ` | 40.0 | 37.5
`ria_mbart` | 42.8 | 25.5
`gazeta_mbart` | 32.6 | 28.2
`rut5_base_sum_gazeta` | 32.2 | 28.1
`mT5-multilingual-XLSum` | 32.2164 | 26.1689


## Dataset

A small part of the MLSUM dataset (200 rows) was modified to collect a set for solution evaluation. The result of summarization was put in the column 'summary', the original summary was removed.
###Dataset Summary

MLSUM is a large-scale MultiLingual SUMmarization dataset, obtained from online newspapers. 
It contains 1.5M+ article/summary pairs in five different languages -- namely, French, German, Spanish, Russian, Turkish. 

[https://aclanthology.org/2020.emnlp-main.647/]


### MLSUM Ru Part

The Russian part was used for the project.

Train | Val | Test
---|---|---
25556  | 750 | 757



### Prerequisites
```
pip install -r requirements.txt
```
#### modify_dataset.py

There are several 

| Argument | Long   |    Description                 | Default
|:----|:------------|-------------------------------|------------------
| -d   | --dev_mode | fast fun on a set of 30 samples | False
| -s   | --slice| number of rows for a file | 50
| -n   | --number | total number of rows to modify | 200


### Results

After test set evaluation the following scores were calculated.

Model | ROUGE-1 | ROUGE-L
---|---|---
`Clustering model ` | 35.9 | 26.7



### Run application

A demo to try extractive summarization was created. 
To run the web part (no model connected yet) run *run.sh* from the *app* folder: `cd app && ./run.sh`. 
 The web interface will be available on *localhost:8080* in your browser.
