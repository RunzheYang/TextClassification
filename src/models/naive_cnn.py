import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NaiveCnn(nn.Module):

	def __init__(self):
		super(NaiveCnn, self).__init__()
		# - 1 input channel; 6 output channels; 4*20
		self.Conv1 = nn.Conv2d(1, 4, (4, 20))
		self.MaxPool1 = nn.MaxPool2d((20, 20))
		self.Conv2 = nn.Conv2d(4, 3, (3, 19))
		self.MaxPool2 = nn.MaxPool2d((5, 1))
		# 2-classification
		self.fc1 = nn.Linear(12, 128)
		self.fc2 = nn.Linear(128, 2)


	def forward(self, x):
		x = self.MaxPool1(F.relu(self.Conv1(x)))
		x = self.MaxPool2(F.relu(self.Conv2(x)))
		x = x.view(-1, 12)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x)

# net = NaiveCnn()
# print net

