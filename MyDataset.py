import os
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None, gray_scale=False):
        self.transform = transform
        self.folder_path = data_path
        self.img_names = os.listdir(self.folder_path)
        self.gray_scale = gray_scale

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        sample = self.img_names[idx]
        sample = Image.open(os.path.join(self.folder_path, sample))
        if self.gray_scale:
            sample = sample.convert('L')
        if self.transform:
            sample = self.transform(sample)

        return sample, idx

    def getImage(self, file_name):
        sample = Image.open(os.path.join(self.folder_path, file_name))
        if self.gray_scale:
            sample = sample.convert('L')
        if self.transform:
            sample = self.transform(sample)
        return sample

    def getPair(self, idx):
        clean_path = ""
        blur_path = ""
        files = os.listdir(clean_path)
        files = sorted(files)
        blur = Image.open(os.path.join(blur_path, files[idx]))
        sample = Image.open(os.path.join(clean_path, files[idx]))
        if self.transform:
            sample = self.transform(sample)
            blur = self.transform(blur)
        return blur, sample