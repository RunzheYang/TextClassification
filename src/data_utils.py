import torch
import h5py
import numpy as np
from torch.utils.data import TensorDataset
from gensim.models import Word2Vec

class Dataset(object):
	
	def __init__(self, val=False):
		print "loading dictionary..."
		self.dict = [0 for i in xrange(378871)]
		with open("support_data/word_index.txt", 'r') as dic_file:
			for line in open("support_data/word_index.txt"):
				rec = dic_file.readline()
				rec = rec[:-1]
				rec = rec.split(' ')
				self.dict[int(rec[0])] = rec[1].decode('utf-8')
		
		print "loading word vectors..."
		self.wv_model = Word2Vec.load("support_data/corpus2vec.model")
		print self.wv_model
		
		print "loading train data..."
		train_data = h5py.File("dataset/train.h5", 'r')
		if val is False:
			self.train_id = train_data['id'][:]
			train_title = train_data['title_feature'][:]
			train_content = train_data['content_feature'][:]
			features = torch.from_numpy(
				np.concatenate((train_title, train_content), axis=1))
			train_label = torch.from_numpy(train_data['label'][:])
			self.trainset = TensorDataset(features, train_label)
		else: # make both training & validation set (last 1000 articles)
			_id = train_data['id'][:]
			tot = len(_id)
			# spl = int(tot*4./5.)
			spl = tot - 1000
			_title = train_data['title_feature'][:]
			_content = train_data['content_feature'][:]
			_features = torch.from_numpy(
				np.concatenate((_title, _content), axis=1))
			_label = torch.from_numpy(train_data['label'][:])
			print _features.size(), _label.size()
			self.trainset = TensorDataset(
				_features[0:spl], _label[0:spl])
			self.valset = TensorDataset(
				_features[spl:tot], _label[spl:tot])
	
	
	def convert2vec(self, inds):
		res = [0 for i in xrange(len(inds))]
		for i in xrange(len(inds)):
			wd = inds[i]
			if wd != -1:
				res[i] = self.wv_model[self.dict[wd]]
			else:
				res[i] = np.zeros(400)
		return np.asarray(res)


	def semanticImg(self, inds):
		return np.asarray(list(map(self.convert2vec, inds)))
		