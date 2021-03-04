from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from pdb import set_trace

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()
		# input_size = 224 x 224
		self.alexnet_rgb = models.alexnet(pretrained=True)
		self.alexnet_depth = models.alexnet(pretrained=True)
		self.alexnet_ir = models.alexnet(pretrained=True)

		# self.fc_rgb = nn.Linear(4096, 2)
		# self.fc_depth = nn.Linear(4096, 2)
		# self.fc_ir = nn.Linear(4096, 2)
		self.fc_combined = nn.Linear(1000 * 3, 2)

	def forward(self, image_rgb, image_depth, image_ir):
		output_rgb = self.alexnet_rgb(image_rgb)
		# output_rgb = self.fc_rgb(output_rgb)

		output_depth = self.alexnet_depth(image_depth)
		# output_depth = self.fc_depth(output_depth)

		output_ir = self.alexnet_ir(image_ir)
		# output_ir = self.fc_ir(output_ir)

		output_combined = torch.cat((output_rgb, output_depth, output_ir), dim=1)
		output_combined = self.fc_combined(output_combined)

		return output_combined

class MaximumMeanDiscrepancy(nn.Module):
	def __init__(self):
		super(MaximumMeanDiscrepancy, self).__init__()

	def forward(self, source, target):
		assert source.size() == target.size()
		
		if (len(source.size())==1):
			source = torch.unsqueeze(source, 0)
			target = torch.unsqueeze(target, 0)

		return torch.sum((torch.mean(source, 0)-torch.mean(target, 0))**2)


class SimilarityEmbedding(nn.Module):
	def __init__(self):
		super(SimilarityEmbedding, self).__init__()

	def forward(self, source, target, label_source, label_target):
		assert source.size() == target.size()
		assert len(source) == len(label_source)
		
		if (len(source.size())==1):
			source = torch.unsqueeze(source, 0)
			target = torch.unsqueeze(target, 0)

		loss = 0

		for i in range(len(source)):
			for j in range(len(target)):
				result = (source[i, :] @ target[j, :])/(torch.sum(source[i, :] ** 2) * torch.sum(target[i, :] ** 2))
				if label_source[i] == label_target[j]:
					loss += 1 - result
				else:
					loss += result if result > 0 else 0

		return loss