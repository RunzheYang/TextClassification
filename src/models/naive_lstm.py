import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NaiveLSTM(nn.Module):

	def __init__(self):
		super(NaiveLSTM, self).__init__()
		# 2-classification
		self.lstm = nn.LSTM(400, 64)
		self.hidden = self.init_hidden()
		self.fc = nn.Linear(64, 2)


	def init_hidden(self, btc=64):
		# (num_layers, minibatch_size, hidden_dim)
		return (Variable(torch.zeros(1, btc, 64)), 
			Variable(torch.zeros(1, btc, 64)))


	def forward(self, x):
		# (len of sequence, mini batch, embadding size)
		x, self.hidden = self.lstm(x.view(520, -1, 400))
		x = self.fc(x[-1].view(-1, 64))
		return F.softmax(x)

# net = NaiveLSTM()
# print net

