import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class StackNN(nn.Module):

	def __init__(self):
		super(StackNN, self).__init__()
		# 8 models 2-classification
		self.fc1 = nn.Linear(16, 256)
		self.fc2 = nn.Linear(256, 2)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return F.softmax(x)

# net = StackNN()
# print net

