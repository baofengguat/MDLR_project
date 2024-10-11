from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np


transformTransfer = transforms.Compose([transforms.ToTensor(),])

class MyDataset(Dataset):
    def __init__(self, names_file, transform=True):
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split('*')[0]
        if not os.path.isfile(image_path):
            print(image_path + ' ' + 'does not exist!')
            return None
        image = Image.open(image_path)
        label = int(self.names_list[idx].split('*')[2])
        PatienceName = os.path.dirname(image_path)

        if self.transform is not None:
            image = transformTransfer(image)
        return image, label,PatienceName


class NEWDataset(Dataset):
    def __init__(self, data,label, transform=None):
        self.data = np.load(data)
        self.label = np.load(label)
        self.transforms = transform

    def __getitem__(self, index):
        image= self.data[index, :, :, :]
        label = self.label[index]

        image = Image.fromarray(image)
        image = transformTransfer(image)

        return image, int(label)

    def __len__(self):
        return self.data.shape[0]

def load_npy(args,dataPath,labelPath):
    train_dataset = NEWDataset(data = dataPath ,
                               label= labelPath)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_loader


def get_readData(args):

    train_dataset = MyDataset(names_file=os.path.join(args.txt_dir,'train_log.txt'),transform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    test_dataset = MyDataset(names_file=os.path.join(args.txt_dir,'test_log.txt'),transform=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader,test_loader






