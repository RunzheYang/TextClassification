# word2vec via gensis

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = "support_data/corpus.txt"
out_file = "support_data/corpus2vec.model"

model =  Word2Vec(
	LineSentence(corpus), 
	size=400, 
	window=5,
	min_count=5, 
	workers=4)

model.save(out_file)
