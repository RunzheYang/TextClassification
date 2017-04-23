import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class FastNet(nn.Module):

	def __init__(self):
		super(FastNet, self).__init__()
		# input (batch, index_stream)
		self.embedding = nn.Embedding(11174, 100)
		self.Conv = nn.Conv2d(1, 2, (4, 100))
		self.MaxPool = nn.MaxPool2d((20, 1))
		# 2-classification
		self.fc = nn.Linear(148, 2)


	def forward(self, x):
		x = x + 1
		x = self.embedding(x)
		x = self.MaxPool(F.relu(self.Conv(x.view(-1, 1, 1500, 100))))
		# print x.size()
		x = x.view(-1, 148)
		x = self.fc(x)
		return F.softmax(x)

# net = FastNet()
# print net

