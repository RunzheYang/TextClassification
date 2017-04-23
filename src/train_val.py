import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import Dataset
from torch.utils.data import DataLoader
# from models.naive_cnn import NaiveCnn
# from models.shallow_cnn import ShallowCnn
from models.naive_lstm import NaiveLSTM

dataset = Dataset(val=True)
# load training data
train_loader = DataLoader(
	dataset.trainset, 
	batch_size=64,
	shuffle=True,
	num_workers=4)
# load validation data
val_loader = DataLoader(
	dataset.valset, 
	batch_size=1,
	shuffle=False,
	num_workers=4)

# load network
net = NaiveLSTM()
net.type(torch.DoubleTensor)

# unbalance data
unbalance = 11.9
# use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# evaluate on validation set by AUC
def eval():
	auc = []
	for i, data in enumerate(val_loader, 0):
		inputs, labels = data
		inputs = torch.unsqueeze(
			torch.from_numpy(dataset.semanticImg(inputs)), 1)
		# wrap them in Variable
		inputs = Variable(inputs)
		# forward
		outputs = net(inputs)
		auc.append([outputs.data[0, 1], labels[0]])
		# if i % 100 == 99: print i+1, "articles in val set have been evaluated."
	auc = np.array(auc)
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


# loop over the dataset multiple times
print "Start Training!"
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
		# get the inputs
		inputs, labels = data
		
		# # cnn input
		# inputs = torch.unsqueeze(
		# 	torch.from_numpy(dataset.semanticImg(inputs)),1)
		
		# rnn input
		inputs = torch.from_numpy(dataset.semanticImg(inputs))
		
		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)

		# zero the parameter gradients
		optimizer.zero_grad()

		# zero the lstm hidden layer
		net.hidden = net.init_hidden(len(labels))

		# forward + backward + optimize
		outputs = net(inputs)
		
		loss = F.cross_entropy(outputs, labels, weight=torch.DoubleTensor([1.0, unbalance]))
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.data[0]
		if i % 200 == 199:    # print every 200 mini-batches
			print('epoch %d, iter %5d, loss: %.3f' % (epoch+1, (i+1), running_loss / 200))
			print('val auc: %.3f' % eval())
			torch.save(net, "params/shallow_cnn.params")
			running_loss = 0.0

print "Finished Training"
eval()
