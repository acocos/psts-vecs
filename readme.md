## PSTS vectors

The PSTS paraphrase vectors come in two types: PP-BERT and PP-W2V.

The first are 768-dimensional paraphrase embeddings based on BERT.

The second are 300-dimensional paraphrase embeddings based on word2vec skip-gram.

If you're looking for the sentences in PSTS themselves, you can find them [here](https://github.com/acocos/paraphrase-sense-tagged-sentences).

If you use this resource in your work, please cite [this paper](https://www.seas.upenn.edu/~acocos/papers/anne-thesis-final.pdf):

```
@phdthesis{cocos19thesis,
  author       = {Anne O'Donnell Cocos}, 
  title        = {Paraphrase-based Models of Lexical Semantics},
  school       = {University of Pennsylvania},
  year         = 2019,
  month        = 5,
}
```

### Download data

To get the vectors, download [this file](https://s3.amazonaws.com/paraphrase-sense-tagged-sentences/psts-vecs.zip) and unzip to this directory.

### Using the vectors

You can load and use both types using the library `vecwrapper.py`

```python

import vecwrapper

ppbert = vecrwapper.PPVecs()
ppbert.load('pp-bert/consolidated_pp_vecs_pmiweights')

v_bug_microbe = bert.vec('NN', 'bug', 'microbe')

bert.nearest_neighbors(v_bug_microbe)

# array(['NN ||| bug ||| microbe', 'NN ||| bug ||| germ',
#        'NN ||| bug ||| bacterium', 'NN ||| bug ||| virus',
#        'NN ||| bug ||| thing', 'NN ||| bug ||| microorganism',
#        'NN ||| bug ||| insect', 'NN ||| bug ||| rat',
#        'NN ||| bug ||| beetle', 'NN ||| bug ||| beast'], dtype='|S52')

bert.similarity('NN', 'bug', 'computer', method='max')
# 0.5950588817627771
bert.similarity('NN', 'bug', 'truck', method='max')
# 0.5042375568708046
```