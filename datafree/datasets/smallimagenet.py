# Reference: https://github.com/landskape-ai/ImageNet-Downsampled
import os
import pickle

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class SmallImagenet(VisionDataset):
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
