import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import sparse_loss, continuity_loss

class double_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class double_conv_up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(double_conv_up, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2):
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])

		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x

# UNet with skip connection and unpooling with pooling indexes
class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.inc = double_conv(3, 32)

		self.conv_down1 = double_conv(32, 64)
		self.pool_down1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 244 -> 122
		self.conv_down2 = double_conv(64, 128)
		self.pool_down2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 122 -> 61
		self.conv_down3 = double_conv(128, 256)
		self.pool_down3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)  # 61 -> 31

		self.bottleneck = double_conv(256, 256)

		self.unpool_up1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 31 -> 61
		self.conv_up1 = double_conv_up(384, 128)
		self.unpool_up2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 61 -> 122
		self.conv_up2 = double_conv_up(192, 64)
		self.unpool_up3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 122 -> 244
		self.conv_up3 = double_conv_up(96, 32)

		self.outc = double_conv(32, 1)

	def forward(self, x):
		# down_sample
		x1 = self.inc(x)
		x2, unpool_map_1 = self.pool_down1(self.conv_down1(x1))
		x3, unpool_map_2 = self.pool_down2(self.conv_down2(x2))
		x4, unpool_map_3 = self.pool_down3(self.conv_down3(x3))

		# bottleneck
		x5 = self.bottleneck(x4)

		# up_sample
		x = self.conv_up1(self.unpool_up1(x5, unpool_map_3), x3)
		x = self.conv_up2(self.unpool_up2(x, unpool_map_2), x2)
		x = self.conv_up3(self.unpool_up3(x, unpool_map_1), x1)
		x = self.outc(x)
		return x


class Losses(nn.Module):  # unet + eblock
	def __init__(self):
		super(Losses, self).__init__()
		self.sparse_loss = sparse_loss()
		self.continuity_loss = continuity_loss()

	def forward(self, x):
		self.sparse_loss(x)
		self.continuity_loss(x)
		return x

class UpCi2(nn.Module):
	def __init__(self, out_classes, rs, modes, down_conv):
		super(UpCi2, self).__init__()
		self.modes = modes
		self.down_conv = down_conv

		self.out_classes = out_classes

		self.resnet_1 = rs[0]
		self.resnet_2 = rs[1]
		self.resnet_3 = rs[2]
		self.resnet_4 = rs[3]

		self.losses = Losses()

		if self.down_conv:
			self.down1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=4, padding=5, dilation=2)
			self.down2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
			self.down3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
		else:
			self.down1 = nn.MaxPool2d(kernel_size=4)
			self.down2 = nn.MaxPool2d(kernel_size=2, padding=1)
			self.down3 = nn.MaxPool2d(kernel_size=2, padding=1)
		self.unet = UNet()

		self.linear = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512, 512),
		    nn.Dropout(),
			nn.Linear(512, self.out_classes, bias=True)
		)

	def forward(self, x):
		y = x.clone()
		y = self.unet(F.relu(y, inplace=True))
		y = self.losses(y)
		y1 = self.down1(y)
		y2 = self.down2(y1)
		y3 = self.down3(y2)

		x = self.resnet_1(x)
		if self.modes[0] == 1:
			x = x * y1
		x = self.resnet_2(x)
		if self.modes[1] == 1:
			x = x * y2
		x = self.resnet_3(x)
		if self.modes[2] == 1:
			x = x * y3
		x = self.resnet_4(x)
		x = self.linear(x)
		return x
