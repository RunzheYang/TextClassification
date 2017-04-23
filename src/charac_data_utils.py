import torch
import h5py
import numpy as np
from torch.utils.data import TensorDataset

class Dataset(object):
	
	def __init__(self, val=False, test=False):
		print "loading train data..."
		train_data = h5py.File("dataset/charac_train.h5", 'r')
		if val is False and test is False:
			self.train_id = train_data['id'][:]
			features = train_data['feature'][:]
			features = torch.from_numpy(features)
			train_label = torch.from_numpy(train_data['label'][:])
			self.trainset = TensorDataset(features, train_label)
		elif test is False: # make both training & validation set (last 1000 articles)
			_id = train_data['id'][:]
			tot = len(_id)
			# spl = int(tot*4./5.)
			spl = tot - 1000
			_features = train_data['feature'][:]
			_features = torch.from_numpy(_features)
			_label = torch.from_numpy(train_data['label'][:])
			# print _features.size(), _label.size()
			self.trainset = TensorDataset(
				_features[0:spl], _label[0:spl])
			self.valset = TensorDataset(
				_features[spl:tot], _label[spl:tot])
		elif test is True:
			test_data = h5py.File("dataset/charac_test.h5", 'r')
			self.test_id = test_data['id'][:]
			features = test_data['feature'][:]
			features = torch.from_numpy(features)
			self.testset = TensorDataset(features, torch.zeros(features.size()))
