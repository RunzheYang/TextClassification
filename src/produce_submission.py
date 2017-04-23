import csv
import torch
from torch.autograd import Variable
from charac_data_utils import Dataset
from torch.utils.data import DataLoader

dataset = Dataset(test=True)

# load test data
test_loader = DataLoader(
	dataset.testset, 
	batch_size=64,
	shuffle=False,
	num_workers=4)

integration_model = torch.load("params/integration_hope_best_up.params")

# load networks
nets = [
	# torch.load("params/fast_net_best.params"), # 0.868829
	# torch.load("params/fast_mul_net_best.params"), # 0.869773
	# torch.load("params/cnn_lstm.params"), # 0.853858
	# torch.load("params/large_cnn_lstm.params"), # 0.879736 
	# torch.load("params/large_cnn_lstm_2.params"), # 0.880812
	torch.load("params/large_cnn_lstm_3.params"), # 0.881556
	torch.load("params/huge_step_1.params"), # 0.890377
	torch.load("params/huge_step_2.params"), # 0.893592
	torch.load("params/huge_step_3.params"), #
	torch.load("params/huge_step_4.params"), # 0.894...
	torch.load("params/ex_huge.params"), #
	torch.load("params/last_hope.params"), # 0.899822
	torch.load("params/last_hope_up.params") # 0.900659
	]

for net in nets:
	for params in net.parameters():
		params.requires_grad = False

for params in integration_model.parameters():
		params.requires_grad = False

def eval(data_set):
	auc = None
	for i, data in enumerate(data_set, 0):
		inputs, labels = data
		# wrap them in Variable
		inputs = Variable(inputs)

		# zero the lstm hidden layer
		for ind in xrange(8):
			nets[ind].hidden = nets[ind].init_hidden(len(labels))

		mul_inputs = torch.cat([nets[ind](inputs) for ind in xrange(8)], 1)

		# forward
		outputs = integration_model(mul_inputs)
		pos_prediction = outputs.data[:, 1]
		sub_auc = pos_prediction
		if auc is None:
			auc = sub_auc
		else:
			auc = torch.cat((auc, sub_auc), 0)
		# print "current auc matrix:", auc
		# if i % 100 == 99: print (i+1)*64*100, "articles have been evaluated."
		print (i+1)*64, "articles have been evaluated."
	# auc = auc.numpy()
	# auc = auc[auc[:,0].argsort()]
	# print auc
	# pos_cnt, neg_cnt, rank, sum_rank = 0, 0, len(auc), 0
	# for rec in auc[::-1]:
	# 	if rec[1] == 0:
	# 		neg_cnt += 1
	# 	elif rec[1] == 1:
	# 		pos_cnt += 1
	# 		sum_rank += rank
	# 	rank -= 1
	# score = (sum_rank - pos_cnt*(pos_cnt+1)/2.0)/(pos_cnt * neg_cnt)
	# return score
	return auc

pred = eval(test_loader)

with open('submissions/integration_hope.csv', 'w') as csvfile:
	fieldnames = ['id', 'pred']
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for ind in xrange(len(dataset.test_id)):
		writer.writerow({'id': dataset.test_id[ind] ,'pred': pred[ind]})

