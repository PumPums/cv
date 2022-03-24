import os
import json
from collections import OrderedDict

from tqdm import tqdm_notebook
from tqdm.autonotebook import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

from .models import UpCi

def get_model_UpSi(after_layer='after_layer2', out_classes=2, device=None):
	eblock_place = {'after_layer1': (61, 5),  # (map_size, resnet_layers_ place)
					'after_layer2': (31, 6),
					'after_layer3': (16, 7)}
	return build_model(eblock_place[after_layer], out_classes, device)


def build_model(params, out_classes, device):
	map_size, k = params
	resnet18 = models.resnet18(pretrained=True)
	for itm in resnet18.parameters():
		itm.requires_grad = False
	resnet_layers = {'conv1': resnet18.conv1, 'bn1': resnet18.bn1, 'relu': resnet18.relu,
					 'maxpool': resnet18.maxpool, 'layer1': resnet18.layer1, 'layer2': resnet18.layer2,
					 'layer3': resnet18.layer3, 'layer4': resnet18.layer4, 'avgpool': resnet18.avgpool}
	resnet_keys_list = list(resnet_layers.keys())
	resnet_1_dct, resnet_2_dct = OrderedDict(), OrderedDict()
	for key in resnet_keys_list[:k]:
		resnet_1_dct.update({key: resnet_layers[key]})
	for key in resnet_keys_list[k:]:
		resnet_2_dct.update({key: resnet_layers[key]})
	resnet_1, resnet_2 = nn.Sequential(resnet_1_dct), nn.Sequential(resnet_2_dct)
	model = UpCi(input_size=map_size, out_classes=out_classes, rs=[resnet_1, resnet_2])
	return model.to(device)


def save_logs(losses, accuracy, name):
	if 'logs' not in os.listdir():
		os.mkdir('logs')
	if name not in os.listdir('logs'):
		os.mkdir('logs/' + name)
	for dct_name, dct in zip(['losses', 'accuracy'], [losses, accuracy]):
		with open('logs/' + name + '/' + dct_name + '.json', 'w') as f:
			json.dump(dct, f)


def train_model_UpSi(model, device, dataloaders, losses_weights=[1, 1, 1], num_epochs=15, modes=['unet', 'linear'],
					 save_log=True, name='test', save_model=1, show_tests=False, test_data=None):
	"""
	model: model
	device: torch.device()
	dataloaders: dict(train_loader, test_loader)
	losses_weights:  class_loss, contiguity_loss, sparsity_loss
	num_epochs: max_epoches
	modes: layers for optimizer
	save_log: True/False
	name: name for saves
	save_model: 0 -> None, 1 -> save best, 2 -> save every epjch
	show_tests: True/False show results of test_data
	test_data: test_data -> list(tensor(1,3,H,W))
	"""
	optimizers = {'resnet_1': torch.optim.Adam(model.resnet_1.parameters(), weight_decay=0.01),
				  'resnet_2': torch.optim.Adam(model.resnet_2.parameters(), weight_decay=0.01),
				  'unet': torch.optim.Adam(model.unet_plus.parameters(), lr=0.0005, weight_decay=0.01),
				  'linear': torch.optim.Adam(model.linear.parameters(), weight_decay=0.01), }
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizers['unet'], step_size=5, gamma=0.25)
	loss_fn = torch.nn.CrossEntropyLoss()

	losses = {'train': {}, 'test': {}}
	accuracy = {'train': {}, 'test': {}}
	res_model = 0

	for epoch in trange(num_epochs, desc=f"All train process"):
		# scheduler.step()
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			total = 0
			correct = 0

			for X_batch, y_batch in tqdm_notebook(dataloaders[phase], leave=False,
												  desc=f"Epoch ({phase})- {epoch + 1}"):
				X_batch = X_batch.to(device)
				y_batch = y_batch.to(device)
				if phase == 'train':
					for mod in modes:
						optimizers[mod].zero_grad()

				if phase == 'train':
					y_pred = model(X_batch)
				else:
					with torch.no_grad():
						y_pred = model(X_batch)

				preds = torch.argmax(y_pred, -1)
				total += y_batch.size(0)
				correct += (preds == y_batch).sum().item()

				class_loss = loss_fn(y_pred, y_batch) * losses_weights[0]
				cont_loss = model.unet_plus.continuity_loss.continuity_loss * losses_weights[1]
				spars_loss = model.unet_plus.sparse_loss.sparse_loss * losses_weights[2]

				loss = class_loss + cont_loss + spars_loss

				if phase == 'train':
					loss.backward()
					for mod in modes:
						optimizers[mod].step()

				losses[phase][epoch] = [float(class_loss), float(cont_loss), float(spars_loss)]
			accuracy[phase][epoch] = float(correct / total)

		print(f'{name} - Epoch-{epoch + 1} {phase} loss:{sum(losses[phase][epoch]):.3f} <->  accuracy: {accuracy[phase][epoch]:.3f}')
		print(f'class_loss:{losses[phase][epoch][0]:.3f}, contiguity_loss:{losses[phase][epoch][1]:.3f}, sparsity_loss:{losses[phase][epoch][2]:.3f}')

		# save_log
		if save_log:
			save_logs(losses, accuracy, name)

		# save_model
		if save_model == 1:
			loss_sum = sum(losses['test'][epoch])
			if epoch > 3:
				if loss_sum < res_model:
					torch.save(model.state_dict(), './logs/' + name + '/' + name + '_best_model.pth')
					res_model = loss_sum
			else:
				res_model = loss_sum
				torch.save(model.state_dict(), './logs/' + name + '/' + name + '_best_model.pth')
		elif save_model == 2:
			torch.save(model.state_dict(), './logs/' + name + '/' + name + '_epoch' + str(epoch) +'.pth')

		# show_tests
		if show_tests and test_data:
			for itm in test_data:
				get_eval_mask([model], itm, device, mode="show", epoch=epoch)
	return model, losses, accuracy


def evaluate(model, device, test_loader, name='test'):
	model.eval()

	total = 0
	correct = 0

	for X_batch, y_batch in tqdm_notebook(test_loader, leave=False, desc=f"Eval:"):
		with torch.no_grad():
			y_pred = model(X_batch.to(device))

		preds = torch.argmax(y_pred, -1)
		total += y_batch.size(0)
		correct += (preds == y_batch).sum().item()

	accuracy = correct / total
	print(f'{name} accuracy = {accuracy}')
	return accuracy

def get_model_resnet18(out_classes):
    resnet18 = models.resnet18(pretrained=True)
    for itm in resnet18.parameters():
        itm.requires_grad = False
    resnet18.fc = nn.Sequential(nn.Flatten(),
                    nn.Linear(512, 512),
                    nn.Dropout(),
                    nn.Linear(512, out_classes, bias=True)
                )
    return resnet18


def train_model_resnet18(model, device, dataloaders, num_epochs=15, save=True, name='resnet18'):
	optimizer = torch.optim.Adam(model.fc.parameters(), weight_decay=0.01)

	loss_fn = torch.nn.CrossEntropyLoss()

	losses = {'train': {}, 'test': {}}
	accuracy = {'train': {}, 'test': {}}
	res_model = 0

	for epoch in trange(num_epochs, desc=f"All train process"):
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			total = 0
			correct = 0

			for X_batch, y_batch in tqdm_notebook(dataloaders[phase], leave=False,
												  desc=f"Epoch ({phase})- {epoch + 1}"):
				X_batch = X_batch.to(device)
				y_batch = y_batch.to(device)
				if phase == 'train':
					optimizer.zero_grad()
				if phase == 'train':
					y_pred = model(X_batch)
				else:
					with torch.no_grad():
						y_pred = model(X_batch)

				preds = torch.argmax(y_pred, -1)
				total += y_batch.size(0)
				correct += (preds == y_batch).sum().item()

				loss = loss_fn(y_pred, y_batch)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				losses[phase][epoch] = float(loss)
				accuracy[phase][epoch] = float(correct / total)

		print(f'{name} - Epoch-{epoch + 1} {phase} loss:{losses[phase][epoch]:.3f} <->  accuracy: {accuracy[phase][epoch]:.3f}')

		if save:
			save_logs(losses, accuracy, name)
		if epoch > 3:
			if losses['test'][epoch] < res_model:
				torch.save(model.state_dict(), './logs/' + name + '/' + name + '_model.pth')
				res_model = losses['test'][epoch]
		else:
			res_model = losses['test'][epoch]
			torch.save(model.state_dict(), './logs/' + name + '/' + name + '_model.pth')
	return model, losses, accuracy

def get_eval_mask(models, img, device, mode="show", epoch=None): # img -> tensor((1, 1, 244, 244))
    # original
	original = img.squeeze().numpy().transpose((1, 2, 0))
	mean = np.array([0.5, 0.5, 0.5])
	std = np.array([0.5, 0.5, 0.5])
	original = std * original + mean
	original = np.clip(original, 0, 1)

	# mask
	masks = []
	for model in models:
		model.eval()
		unet_part = model.unet_plus.unet
		with torch.no_grad():
			result = unet_part(img.to(device))
		mask = result.detach().cpu().squeeze().numpy()
		masks.append(mask)

	if mode == 'show':
		ncols = len(models)
		fig, ax = plt.subplots(nrows=1, ncols=ncols+1, figsize=(12, 7))
		if epoch:
			ax[0].set_title('epoch: ' + str(epoch), color='g')
		ax[0].imshow(original)
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		ax[0].set_xticklabels([])
		ax[0].set_yticklabels([])
		for i in range(ncols):
			ax[i+1].set_title('model: ' + str(i + 1))
			ax[i+1].imshow(masks[i])
			ax[i+1].set_xticks([])
			ax[i+1].set_yticks([])
			ax[i+1].set_xticklabels([])
			ax[i+1].set_yticklabels([])
	else:
		return masks