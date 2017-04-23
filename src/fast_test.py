import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from charac_data_utils import Dataset
from torch.utils.data import DataLoader
# from models.fast_net import FastNet
from models.fast_multikernel_net import FastMulNet


dataset = Dataset(val=True)

# load test data
val_loader = DataLoader(
	dataset.valset, 
	batch_size=64,
	shuffle=False,
	num_workers=4)

nets = [
	torch.load("params/fast_net_best.params"), # 0.868829
	torch.load("params/fast_mul_net_best.params"), # 0.869773
	torch.load("params/cnn_lstm.params"), # 0.853858
	torch.load("params/large_cnn_lstm.params"), # 0.879736
	torch.load("params/large_cnn_lstm_2.params"), # 0.880812
	torch.load("params/large_cnn_lstm_3.params"), # 0.881556
	torch.load("params/huge_step_1.params"), # 0.890377
	torch.load("params/huge_step_2.params") # 0.893592
	]


def eval(data_loader, net):
	auc = None
	for i, data in enumerate(data_loader, 0):
		inputs, labels = data
		# wrap them in Variable
		inputs = Variable(inputs)
		# zero the lstm hidden layer
		if net.__class__.__name__ != "FastNet" and net.__class__.__name__ != "FastMulNet":
			net.hidden = net.init_hidden(len(labels))
		# forward
		outputs = net(inputs)
		pos_prediction = outputs.data[:, 1]
		sub_auc = torch.cat((pos_prediction, labels.type(torch.FloatTensor)), 1)
		if auc is None:
			auc = sub_auc
		else:
			auc = torch.cat((auc, sub_auc), 0)
		# print "current auc matrix:", auc
		if i % 100 == 99: print (i+1)*100, "articles have been evaluated."
	auc = auc.numpy()
	auc = auc[auc[:,0].argsort()]
	print auc
	pos_cnt, neg_cnt, rank, sum_rank = 0, 0, len(auc), 0
	for rec in auc[::-1]:
		if rec[1] == 0:
			neg_cnt += 1
		elif rec[1] == 1:
			pos_cnt += 1
			sum_rank += rank
		rank -= 1
	score = (sum_rank - pos_cnt*(pos_cnt+1)/2.0)/(pos_cnt * neg_cnt)
	return score

for net in nets:
	print  net.__class__.__name__,'overall auc: %.6f' % eval(val_loader, net)
