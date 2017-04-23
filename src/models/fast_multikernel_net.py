import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class FastMulNet(nn.Module):

	def __init__(self):
		super(FastMulNet, self).__init__()
		# input (batch, index_stream)
		self.embedding = nn.Embedding(11174, 200)
		self.Conv1 = nn.Conv2d(1, 2, (2, 200))
		self.MaxPool1 = nn.MaxPool2d((20, 1))
		self.Conv2 = nn.Conv2d(1, 2, (3, 200))
		self.MaxPool2 = nn.MaxPool2d((20, 1))
		self.Conv3 = nn.Conv2d(1, 2, (4, 200))
		self.MaxPool3 = nn.MaxPool2d((20, 1))
		# 2-classification
		self.fc = nn.Linear(444, 2)


	def forward(self, x):
		x = x + 1
		x = self.embedding(x)
		feamap1 = self.MaxPool1(F.relu(self.Conv1(x.view(-1, 1, 1500, 200))))
		feamap2 = self.MaxPool2(F.relu(self.Conv2(x.view(-1, 1, 1500, 200))))
		feamap3 = self.MaxPool3(F.relu(self.Conv3(x.view(-1, 1, 1500, 200))))
		x = torch.cat((feamap1, feamap2, feamap3), 1)
		# print x.size()
		x = x.view(-1, 444)
		x = self.fc(x)
		return F.softmax(x)

# net = FastMulNet()
# print net

