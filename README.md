# A web application for extractive and abstractive summarization 

## Introduction

This is a project for Natural Language Processing course (ods.ai, Huawei, december 2021)
**Extractive summarization of the news articles.**

The project consists of two parts:
* Creation of extractive summaries from the text 
* Using these summaries to fine-tune a language model.




There are several models, that were fine-tuned on Russian new datasets for summarization:

Model | ROUGE-1 | ROUGE-L
---|---|---
`ria_copynet_10kk ` | 40.0 | 37.5
`ria_mbart` | 42.8 | 25.5
`gazeta_mbart` | 32.6 | 28.2
`rut5_base_sum_gazeta` | 32.2 | 28.1
`mT5-multilingual-XLSum` | 32.2164 | 26.1689


## Dataset

###Dataset Summary

MLSUM is a large-scale MultiLingual SUMmarization dataset, obtained from online newspapers. 
It contains 1.5M+ article/summary pairs in five different languages -- namely, French, German, Spanish, Russian, Turkish. 

[https://aclanthology.org/2020.emnlp-main.647/]


### MLSUM Ru Part

The Russian part was used for the project.

Train | Val | Test
---|---|---
25556  | 750 | 757

In the project the modification of the original dataset was proposed.
Similar to the Pegasus model training, there is a module to create 3 sentence extractive summaries instead of 1 sentence original human-written summary, which are stored in the 'summary' column of the dataset.

The summaries in the dataset are replaced by extractive summaries, obtained from the **K-Means clustering** model.

More about **Pegasus**:
https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html

https://arxiv.org/abs/1912.08777

The aim of this part is to teach the model take relevant parts of the original article rather than generate new text.

In the project the training on the original and modified datasets are supported.

### Language model

The second part of the project is build on fine-tuning *t5-small* checkpoint[https://arxiv.org/abs/1910.10683]
 as shown in the **Hugging Face** tutorial:
https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb


Model | Description | # params
---|---|---
`T5-Small ` | t5 model checkpoint | 60M 


## Fine-Tuning

The example below shows how to finetune the `T5-small` model on the MLSUM dataset for summarization.

### Prerequisites
```
pip install -r requirements.txt
```
#### modify_dataset.py


Script for modify MLSUM dataset. 

| Argument | Long   |    Description                 | Default
|:----|:------------|-------------------------------|------------------
| -d   | --dev_mode | fast fun on a set of 30 samples | False
| -s   | --slice| length of temporary files | 50
| -o   | --original | prepare original dataset for training | False


### train.py

For fine-tuning on 1 GPU settings the following hyperparameters might be used:

```
python train.py 
  --batch_size 8 \
  --epochs 15 \
  --save_model True
```


### Results
For the project 3 models were fine-runed:
the Pegasus model was fine-tuned on the Gazeta dataset
It did not require any coding.


Model | ROUGE-1 | ROUGE-L
---|---|---
`autonlp-pegasus-ru-21074422 ` | 9.5923 | 9.4904
`t5small-mlsum-ru` | 2 | 2
`t5small-mlsum-extr-ru` | 2 | 2


### Run application

A demo to try extractive summarization was created. 
To run the web part (no model connected yet) run *run.sh* from the *app* folder: `cd app && ./run.sh`. 
 The web interface will be available on *localhost:8080* in your browser.
