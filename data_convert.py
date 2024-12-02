import os
import configs
import util_functions
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class HRDataset(Dataset):
    def __init__(self, dataset, encoder, mode='train'):
        super().__init__()
        if mode == 'train':
            self.dataset_path = configs.train_dir
        if mode == 'valid':
            self.dataset_path = configs.valid_dir
        if mode == 'test':
            self.dataset_path = configs.test_dir

        self.data = dataset
        self.encoder = encoder

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        #Get file path from csv information
        img_name = self.data.iloc[index]['FILENAME']
        #Read image
        image = cv2.imread(os.path.join(self.dataset_path, img_name),
                           cv2.IMREAD_GRAYSCALE)
        #Normalize image to standard size
        image = cv2.resize(image, configs.image_size,
                           interpolation=cv2.INTER_AREA).rotate(image, cv2.ROTATE_90_CLOCKWISE) / 255.

        image = np.expand_dims(image, 0)

        #Get image label from csv file and encode the information into the image
        img_label = self.data.iloc[index]['IDENTITY']
        img_label = self.encoder(img_label)

        #Convert to tensor
        image = torch.as_tensor(image, dtype=torch.float32)
        img_label = torch.as_tensor(img_label, dtype=torch.int32)

        return image, img_label