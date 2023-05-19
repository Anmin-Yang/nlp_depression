# nlp_depression

This is the repository for the natural language processing of narrative writing for depression screening in adolescents.

## Battery Description

We tested the quality of the battery with Cronbach's Alpha.

See [0.0_c_alpha.py](https://github.com/Anmin-Yang/nlp_depression/blob/main/0.0_c_alpha.py).

## Classic ML Methods

### LIWC as Features

We used LIWC as features and classic ML methods as classifiers.

- [1.1_LIWC_ML_traditional.py](https://github.com/Anmin-Yang/nlp_depression/blob/main/1.1_LIWC_ML_traditional.py)

- [1.2_LIWC_XGB.py](https://github.com/Anmin-Yang/nlp_depression/blob/main/1.2_LIWC_XGB.py)

### Word2Vec as Features

We used pre-trained Word2Vec as features and classic ML methods as classifiers.

- [2.1_word2vec_ML_traditional.py](https://github.com/Anmin-Yang/nlp_depression/blob/main/2.1_word2vec_ML_traditional.py)

- [2.2_word2vec_XGB.py](https://github.com/Anmin-Yang/nlp_depression/blob/main/2.2_word2vec_XGB.py)

## Neural Networks

We used pre-trained Word2Vec as features, TextCNN and TextRNN as classifiers. 

Training:

- []()