import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from charac_data_utils import Dataset
from torch.utils.data import DataLoader
from models.integration import StackNN

# import itchat

# itchat.login(enableCmdQR=False)

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
nets = [
	torch.load("params/fast_net_best.params"), # 0.868829
	torch.load("params/fast_mul_net_best.params"), # 0.869773
	torch.load("params/cnn_lstm.params"), # 0.853858
	torch.load("params/large_cnn_lstm.params"), # 0.879736 
	torch.load("params/large_cnn_lstm_2.params"), # 0.880812
	torch.load("params/large_cnn_lstm_3.params"), # 0.881556
	torch.load("params/huge_step_1.params"), # 0.890377
	torch.load("params/huge_step_2.params"), # 0.893592
	torch.load("params/huge_step_3.params"), #
	torch.load("params/huge_step_4.params"), #
	torch.load("params/ex_huge.params") #
	]

for net in nets:
	for params in net.parameters():
		params.requires_grad = False


integration_model = StackNN()
# integration_model = torch.load("params/integration_final.params")

unbalance = torch.FloatTensor([1.0, 4.8])
# use Adam optimizer
optimizer = optim.Adam(integration_model.parameters(), lr=2e-4)

# evaluate on validation set by AUC
def eval():
	auc = None
	for i, data in enumerate(val_loader, 0):
		inputs, labels = data
		# wrap them in Variable
		inputs = Variable(inputs)

		# zero the lstm hidden layer
		for ind in range(2, 11):
			nets[ind].hidden = nets[ind].init_hidden(len(labels))

		mul_inputs = torch.cat([nets[ind](inputs) for ind in xrange(11)], 1)

		# forward
		outputs = integration_model(mul_inputs)
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
# print eval()
cur_val_auc = 0
# loop over the dataset multiple times
last_val_auc = 0.90
print "Start Training!"
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
		for ind in range(2, 11):
			nets[ind].hidden = nets[ind].init_hidden(len(labels))
		
		# nets[2].hidden = nets[2].init_hidden(len(labels))
		# nets[3].hidden = nets[3].init_hidden(len(labels))
		# nets[4].hidden = nets[4].init_hidden(len(labels))
		# nets[5].hidden = nets[5].init_hidden(len(labels))

		mul_inputs = torch.cat([nets[ind](inputs) for ind in xrange(11)], 1)

		# forward + backward + optimize
		outputs = integration_model(mul_inputs)
		
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
				torch.save(integration_model, "params/integration_final_3.params")
			# itchat.send(
			# 	"Hi! Here is the latest report\nepoch %d\niter %5d\nloss %.6f\nval auc %.6f\n" % (epoch+1, (i+1), running_loss / 200, cur_val_auc), 
			# 	toUserName='filehelper')
			running_loss = 0.0

print "Finished Training"
print 'val auc: %.6f' % eval()

