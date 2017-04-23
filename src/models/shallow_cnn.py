import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ShallowCnn(nn.Module):

	def __init__(self):
		super(ShallowCnn, self).__init__()
		self.title_ind = Variable(torch.LongTensor(range(0,20)))
		self.content_ind = Variable(torch.LongTensor(range(20,520)))
		# title convolution
		# - 1 input channel; 2 output channels; 2*400, 3*400, 4*400
		self.titleConv1 = nn.Conv2d(1, 2, (2, 400))
		self.titleConv2 = nn.Conv2d(1, 2, (3, 400))
		self.titleConv3 = nn.Conv2d(1, 2, (4, 400))
		self.titleMaxPool1 = nn.MaxPool2d((19, 1))
		self.titleMaxPool2 = nn.MaxPool2d((18, 1))
		self.titleMaxPool3 = nn.MaxPool2d((17, 1))
		# content convolution
		# - 1 input channel; 2 output channels; 2*400, 3*400, 4*400
		self.contentConv1 = nn.Conv2d(1, 2, (2, 400))
		self.contentConv2 = nn.Conv2d(1, 2, (3, 400))
		self.contentConv3 = nn.Conv2d(1, 2, (4, 400))
		self.contentMaxPool1 = nn.MaxPool2d((499, 1))
		self.contentMaxPool2 = nn.MaxPool2d((498, 1))
		self.contentMaxPool3 = nn.MaxPool2d((497, 1))
		# 2-classification
		self.fc1 = nn.Linear(12, 32)
		self.fc2 = nn.Linear(32, 2)


	def forward(self, x):
		title = torch.index_select(x, 2, self.title_ind)
		titleFea1 = self.titleMaxPool1(F.relu(self.titleConv1(title)))
		titleFea2 = self.titleMaxPool2(F.relu(self.titleConv2(title)))
		titleFea3 = self.titleMaxPool3(F.relu(self.titleConv3(title)))

		content = torch.index_select(x, 2, self.content_ind)
		contentFea1 = self.contentMaxPool1(F.relu(self.contentConv1(content)))
		contentFea2 = self.contentMaxPool2(F.relu(self.contentConv2(content)))
		contentFea3 = self.contentMaxPool3(F.relu(self.contentConv3(content)))

		x = torch.cat((titleFea1, titleFea2, titleFea3, contentFea1, contentFea2, contentFea3))
		x = x.view(-1, 12)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x)

# net = ShallowCnn()
# print net

