Code for our NAACL 2019 paper:

## Sentence Embedding Alignment for Lifelong Relation Extraction

Paper link: [https://arxiv.org/abs/1903.02588](https://arxiv.org/abs/1903.02588)

### Running the code
First prepare the GloVe embeddings
* download data from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/), put glove.6B.300d.txt under ``data/``


#### 1. On Simple Question Dataset 
``cp config_SimQue.py config.py``

``python continue_train.py 100``

#### 2. On FewRel Dataset 
``cp config_FewRel.py config.py``

``python continue_train.py 100``
