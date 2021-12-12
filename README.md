# A web application for extractive and abstractive summarization 

## Run application

```bash

- cd app && ./run.sh
```



## Introduction

MBART is a sequence-to-sequence denoising auto-encoder pre-trained on large-scale monolingual corpora in many languages using the BART objective. 
[https://arxiv.org/abs/2001.08210]


## Original model

Model | Description | # params
---|---|---
`mbart.CC25` | mBART model with 12 encoder and decoder layers trained on 25 languages' monolingual corpus | 610M 


## Dataset

###Dataset Summary

We present MLSUM, the first large-scale MultiLingual SUMmarization dataset. Obtained from online newspapers, it contains 1.5M+ article/summary pairs in five different languages -- namely, French, German, Spanish, Russian, Turkish. 
https://aclanthology.org/2020.emnlp-main.647/

For training was used the part with the Russian articles.
### MLSUM Ru Part
Train | Val | Test
---|---|---
25556  | 750 | 757



## Results

Model | ROUGE-1 | ROUGE-L
---|---|---
`mT5-multilingual-XLSum` | 32.2164 | 26.1689
`mbart_ru_sum_gazeta` | 32.4 | 28.0
`mbart-ru-MLSUM` | - | -


