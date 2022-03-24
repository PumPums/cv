import torch
import torch.nn as nn
import torch.nn.functional as F


class sparse_loss(nn.Module):
	def __init__(self):
		super(sparse_loss, self).__init__()
		self.L1Loss = nn.L1Loss(size_average=True)
		self.sparse_loss = 0

	def forward(self, x):
		img = x[:, 0, :, :].clone()  # add for batches
		img = F.relu(img)
		self.sparse_loss = torch.mean(img[:, :, :])
		return x


class continuity_loss(nn.Module):
	def __init__(self):
		super(continuity_loss, self).__init__()
		self.continuity_loss = 0

	def forward(self, x):
		img = x[:, 0, :, :].clone()  # add for batches
		img = F.relu(img)
		loss_x = img[:, 1:, :] - img[:, :-1, :]
		loss_y = img[:, :, 1:] - img[:, :, :-1]
		self.continuity_loss = abs(torch.mean(loss_x)) + abs(torch.mean(loss_y))
		return x