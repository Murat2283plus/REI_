import torch
from torch.utils.data.dataset import Dataset
import scipy.io as scio


import os
import pydicom
from PIL import Image
import numpy as np

def load_and_preprocess_data(data_path):
    file_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.dcm'):
                file_list.append(os.path.join(root, file))
                
    num_images = len(file_list)
    resized_images = np.empty((num_images, 256, 256), dtype=np.float32)
    
    for idx, file_path in enumerate(file_list):
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        img_resized = np.array(Image.fromarray(img).resize((256, 256)))
        resized_images[idx] = img_resized
        
    return resized_images

path = 'dataset/CT/dicom_dir'
x = load_and_preprocess_data(path)
x = torch.from_numpy(x)


class CTData(Dataset):
    """CT dataset."""
    def __init__(self, mode='train', root_dir=path, sample_index=None):
        # the original CT100 dataset can be downloaded from
        # https://www.kaggle.com/kmader/siim-medical-images
        # the images are resized and saved in Matlab.

        # mat_data = scio.loadmat(root_dir)
        # x = torch.from_numpy(mat_data['DATA'])
        path = 'dataset/CT/dicom_dir'
        x = load_and_preprocess_data(path)
        x = torch.from_numpy(x)

        if mode=='train':
             self.x = x[0:90]#murat
            #  self.x = x[0:90]
        if mode=='test':
            self.x = x[90:100,...]#murat
            # self.x = x[50:100]

        self.x = self.x.type(torch.FloatTensor)

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)