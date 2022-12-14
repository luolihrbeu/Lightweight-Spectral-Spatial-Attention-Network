import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
	def __init__(self, dataset, transfor):
		self.data = dataset[0].astype(np.float32)
		self.transformer = transfor
		self.labels = []
		for n in dataset[1]: self.labels += [int(n)]

	def __getitem__(self, index):
		img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
		if self.transformer is not None:
			img = self.transformer(img)
		label = self.labels[index]
		return img, label

	def __len__(self):
		return len(self.labels)

	def __labels__(self):
		return self.labels
