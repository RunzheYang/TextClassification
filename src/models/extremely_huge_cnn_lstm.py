import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ExHugeCnnLSTM(nn.Module):

	def __init__(self):
		super(ExHugeCnnLSTM, self).__init__()
		# input (batch, index_stream)
		self.embedding = nn.Embedding(11174, 128)
		# Convolutional Layers
		self.Conv1 = nn.Conv2d(1, 100, (2, 128))
		self.MaxPool1 = nn.MaxPool2d((10, 1))
		self.Conv2 = nn.Conv2d(1, 100, (3, 128))
		self.MaxPool2 = nn.MaxPool2d((10, 1))
		self.Conv3 = nn.Conv2d(1, 100, (4, 128))
		self.MaxPool3 = nn.MaxPool2d((10, 1))
		self.Conv4 = nn.Conv2d(1, 100, (5, 128))
		self.MaxPool4 = nn.MaxPool2d((10, 1))
		self.Conv5 = nn.Conv2d(1, 100, (6, 128))
		self.MaxPool5 = nn.MaxPool2d((10, 1))
		self.Conv6 = nn.Conv2d(1, 100, (7, 128))
		self.MaxPool6 = nn.MaxPool2d((10, 1))
		# BiLSTM
		self.lstm = nn.LSTM(
			input_size=600,
			hidden_size=72,
			num_layers = 1,
			batch_first=True,
			bidirectional=True)
		self.hidden = self.init_hidden()

		# 2-classification
		self.fc = nn.Linear(288, 2)


	def init_hidden(self, btc=64):
		# (num_layers, minibatch_size, hidden_dim)
		return (Variable(torch.zeros(1, btc, 72)), 
			Variable(torch.zeros(1, btc, 72)))


	def forward(self, x):
		x = x + 1
		x = self.embedding(x)
		feamap1 = self.MaxPool1(F.relu(self.Conv1(x.view(-1, 1, 1500, 128))))
		feamap2 = self.MaxPool2(F.relu(self.Conv2(x.view(-1, 1, 1500, 128))))
		feamap3 = self.MaxPool3(F.relu(self.Conv3(x.view(-1, 1, 1500, 128))))
		feamap4 = self.MaxPool4(F.relu(self.Conv4(x.view(-1, 1, 1500, 128))))
		feamap5 = self.MaxPool5(F.relu(self.Conv5(x.view(-1, 1, 1500, 128))))
		feamap6 = self.MaxPool6(F.relu(self.Conv6(x.view(-1, 1, 1500, 128))))
		# print feamap1.size(), feamap2.size(), feamap3.size(), feamap4.size(), feamap5.size()
		x = torch.cat((feamap1, feamap2, feamap3, feamap4, feamap5, feamap6), 1).squeeze(3)
		x = torch.transpose(x, 1, 2)

		lstm_out, self.hidden = self.lstm(x)
		# print lstm_out
		
		x = torch.cat((lstm_out[:,0], lstm_out[:,-1]), 1)
		

		x = self.fc(x)
		return F.softmax(x)

# net = ExHugeCnnLSTM()
# print net

