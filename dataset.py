from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
class Dataset(Dataset):
    def __init__(self, df,label,transform=None):
        self.df = df
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]  # 假设第一列是图片路径
        label = self.label[idx]    # 假设第二列是标签
        image = Image.open(img_path)  
        if self.transform:
            tensor = self.transform(image)
        return tensor, label

transform = transforms.Compose([
    transforms.ToTensor()
])