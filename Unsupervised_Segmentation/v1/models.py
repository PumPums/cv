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


class UNet_plus(nn.Module):  # unet + eblock
	def __init__(self, input_size):
		super(UNet_plus, self).__init__()
		self.input_size = input_size
		self.unet = UNet()
		self.coefs_dct = self.calculs()
		self.maxpool2d = nn.MaxPool2d(kernel_size=self.coefs_dct['ker_size'],
									  padding=self.coefs_dct['pad'],)
		# losses
		self.sparse_loss = sparse_loss()
		self.continuity_loss = continuity_loss()

	def forward(self, x):
		x = self.unet(F.relu(x, inplace=True))
		x = self.maxpool2d(x)
		self.sparse_loss(x)
		self.continuity_loss(x)
		return x

	def calculs(self, udim=244):
		a1, a2 = divmod(udim, self.input_size)
		if a2 != 0:
			a1 += 1
		b1, b2 = divmod(udim, a1)
		if b2 != 0:
			b1 += 1
		new_udim = b1 * a1
		return {'ker_size': a1, 'pad': (new_udim - udim) // 2}


class UpCi(nn.Module):
	def __init__(self, input_size, out_classes, rs):
		super(UpCi, self).__init__()
		self.input_size = input_size
		self.out_classes = out_classes
		self.resnet_1 = rs[0]  # before_e-block
		self.unet_plus = UNet_plus(self.input_size)
		self.resnet_2 = rs[1]  # after_e-block
		self.linear = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512, 512),
		    nn.Dropout(),
			nn.Linear(512, self.out_classes, bias=True)
		)

	def forward(self, x):
		y = x.clone()
		x = self.resnet_1(x)
		unet_plus_out = self.unet_plus(y)
		unet_plus_out.unsqueeze(dim=1)
		x = x * unet_plus_out
		x = self.resnet_2(x)
		x = self.linear(x)
		return x
