from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CovidLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 2])
        image = io.imread(img_name)
        
        # One Hot Labels
        labels = self.landmarks_frame.iloc[idx, 1]
        labels = labels.split('|')

        gt_label = np.zeros((15),np.float32)
        classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion','Emphysema',
        			'Fibrosis','Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
 					'Pneumothorax', 'COVID-19']
        for i in range(15):
        	for j in labels:
	            if j == classes[i]:
	                gt_label[i] = 1.0
        
	     # Sample

        sample = {'image': image, 'gt_label': gt_label}

        return sample

Covid_dataset = CovidLandmarksDataset(csv_file='./final-dataset/metadata.csv',
                                           root_dir='./final-dataset/images/')

dataloader = DataLoader(Covid_dataset, batch_size=1,
                        shuffle=True, num_workers=2)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['gt_label'])