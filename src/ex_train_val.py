import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from charac_data_utils import Dataset
from torch.utils.data import DataLoader
from models.extremely_huge_cnn_lstm import ExHugeCnnLSTM

# import itchat

# itchat.login(enableCmdQR=True)

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
	batch_size=64,
	shuffle=False,
	num_workers=4)

# load networks
net = ExHugeCnnLSTM()
net = torch.load("params/ex_huge.params")

unbalance = torch.FloatTensor([1.0, 10.8])
# use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=5e-5)

# evaluate on validation set by AUC
def eval():
	auc = None
	for i, data in enumerate(val_loader, 0):
		inputs, labels = data
		# wrap them in Variable
		inputs = Variable(inputs)

		# zero the lstm hidden layer
		net.hidden = net.init_hidden(len(labels))

		# forward
		outputs = net(inputs)
		pos_prediction = outputs.data[:, 1]
		sub_auc = torch.cat((pos_prediction, labels.type(torch.FloatTensor)), 1)
		if auc is None:
			auc = sub_auc
		else:
			# print auc.type()
			# print sub_auc.type()
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

print eval()
cur_val_auc = 0
# loop over the dataset multiple times
print "Start Training!"
last_val_auc = 0.
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
		# get the inputs
		inputs, labels = data
		
		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)

		# zero the parameter gradients
		optimizer.zero_grad()

		# zero the lstm hidden layer
		net.hidden = net.init_hidden(len(labels))

		# forward + backward + optimize
		outputs = net(inputs)
		
		loss = F.cross_entropy(outputs, labels, weight=unbalance)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.data[0]
		print "current loss:", loss.data[0]
		if i % 200 == 199:    # print every 200 mini-batches
			print('epoch %d, iter %5d, loss: %.6f' % (epoch+1, (i+1), running_loss / 200))
			cur_val_auc = eval()
			print('val auc: %.6f' % cur_val_auc)
			if cur_val_auc > last_val_auc: 
				last_val_auc = cur_val_auc
				torch.save(net, "params/ex_huge.params")
			# itchat.send(
			# 	"Hi! Here is the latest report\nepoch %d\niter %5d\nloss %.6f\nval auc %.6f\n" % (epoch+1, (i+1), running_loss / 200, cur_val_auc), 
			# 	toUserName='filehelper')
			running_loss = 0.0

print "Finished Training"
print 'val auc: %.6f' % eval()

